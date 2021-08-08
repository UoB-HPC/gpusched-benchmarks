#include <omp.h>

#include <gpusched.h>

#include "../common.hh"

static team_t* h_team = nullptr;
static team_t* d_team = nullptr;
static __shared__ team_t* device_d_team;

struct cla_count_t {
  int count;
  char pad[CACHELINE_SIZE - sizeof(int)];
};
static_assert(sizeof(cla_count_t) == 128,
              "count struct does not equal cacheline size");

static __shared__ int mycount;
static __device__ int d_total_count = 0;

__device__ static worker_t* get_worker()
{
  return device_d_team->workers + gpu_utils::global_worker_id();
}

#define MAX_SOLUTIONS_INTS \
  ((MAX_SOLUTIONS + sizeof(size_t) - 1) / sizeof(size_t))  // should be 2
#define MAX_SOLUTIONS_R MAX_SOLUTIONS_INTS * sizeof(size_t)

__device__ int ok(int n, char* a)
{
  int i, j;
  char p, q;

  for (i = 0; i < n; i++) {
    p = a[i];

    for (j = i + 1; j < n; j++) {
      q = a[j];
      if (q == p || q == p - (j - i) || q == p + (j - i)) return 0;
    }
  }
  return 1;
}

__device__ void nqueens_ser(int n, int j, char* a)
{
  if (n == j) {
    if (gpu_utils::thread_id() == 0) mycount++;
    //printf("bottom on %d: %d\n", gpu_utils::global_worker_id(), *mycount);
    return;
  }

  for (int i = 0; i < n; ++i) {
    a[j] = (char)i;
    if (ok(j + 1, a)) {
      nqueens_ser(n, j + 1, a);
    }
  }
}

__device__ void nqueens(int n, int j, char* a, int depth);

__device__ void nqueens_task(worker_t* worker, task_t* task)
{
  __shared__ char b[MAX_SOLUTIONS_R];
  size_t* b_ptr = (size_t*)b;

  int k;
  for (k = 0; k < MAX_SOLUTIONS_INTS; ++k) {
    b_ptr[k] = (size_t)get_private(task, k);
  }

  int n = (int)get_private(task, k++);
  int i = (int)get_private(task, k++);
  int j = (int)get_private(task, k++);
  int depth = (int)get_private(task, k++);

  // printf("task on %d (n,i,j,depth) = %d, %d, %d, %d\n",
  // gpu_utils::global_worker_id(), n, i, j, depth);

  b[j] = (char)i;
  if (ok(j + 1, b)) {
    nqueens(n, j + 1, b, depth + 1);
  }
}

__device__ void nqueens(int n, int j, char* a, int depth)
{
  if (n == j) {
    if (gpu_utils::thread_id() == 0) mycount++;
    printf("bottom on %d: %d\n", gpu_utils::global_worker_id(), mycount);
    return;
  }

  worker_t* worker = get_worker();

  for (int i = 0; i < n; ++i) {
    if (depth < CUTOFF) {
      const int nargs = MAX_SOLUTIONS_INTS + 4;
      void* args[nargs];
      size_t* a_ptr = (size_t*)a;
      int k;
      for (k = 0; k < MAX_SOLUTIONS_INTS; ++k) {
        args[k] = (void*)(a_ptr[k]);
      }
      args[k++] = (void*)n;
      args[k++] = (void*)i;
      args[k++] = (void*)j;
      args[k++] = (void*)depth;
      // printf("gen task on %d (n,i,j,depth) = %d, %d, %d, %d\n",
      // gpu_utils::global_worker_id(), n, i, j, depth);
      generate_task(worker, nqueens_task, nargs, args);

      //#pragma omp task
      //      {
      //        char b[MAX_SOLUTIONS_R];
      //        memcpy(b, a, j * sizeof(char));
      //        b[j] = (char)i;
      //        if (ok(j + 1, b)) {
      //          nqueens(n, j + 1, b, depth + 1);
      //        }
      //      }
    }
    else {
      a[j] = (char)i;
      if (ok(j + 1, a)) {
        nqueens_ser(n, j + 1, a);
      }
    }
  }
  //#pragma omp taskwait
  taskwait(worker);
}

__device__ void entry_point(worker_t* worker, task_t* task)
{
  if (gpu_utils::thread_id() == 0) {
    mycount = 0;
    device_d_team = (team_t*)get_private(task, 1);
  }
  gpu_utils::sync_worker();

  int size = (int)get_private(task, 0);

  if (gpu_utils::global_worker_id() == 0) {
    char a[MAX_SOLUTIONS_R];
    nqueens(size, 0, a, 0);
  }
  barrier(worker);

  if (gpu_utils::thread_id() == 0) {
    //printf("worker = %d, count = %d\n", gpu_utils::global_worker_id(),
    //       mycount);
    atomicAdd(&d_total_count, mycount);
  }

  if (gpu_utils::thread_id() == 0 && gpu_utils::global_worker_id() == 0) {
    //for (int i = 0; i < 10; ++i) {
    //  printf("out = %d\n", counts[i].count);
    //}
  }
}

double find_nqueens(int size, int* total_count)
{
  const char* num_blocks_str = getenv("GPUSCHED_NUM_BLOCKS");
  const int num_blocks =
      (num_blocks_str == NULL) ? 56 * 5 : atoi(num_blocks_str);
  create_team(num_blocks, 1024 * 512, &h_team, &d_team);

  const int nargs = 2;
  void* args[nargs] = {(void*)size, (void*)d_team};

  double t0 = omp_get_wtime();
  fork_team<entry_point>(h_team, d_team, nargs, args, num_blocks * NWORKERS);
  double t1 = omp_get_wtime();

  CUDACHK(cudaMemcpyFromSymbol(total_count, d_total_count, sizeof(int), 0,
                               cudaMemcpyDeviceToHost));

  return (t1 - t0);
}
