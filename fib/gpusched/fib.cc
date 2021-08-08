#include <stdio.h>
#include <omp.h>

#include <gpusched.h>

static team_t* h_team = nullptr;
static team_t* d_team = nullptr;

__device__ int64_t d_result;

void init_fib()
{
  const char* num_blocks_str = getenv("GPUSCHED_NUM_BLOCKS");
  const int num_blocks =
    (num_blocks_str == NULL) ? 56 * 5 : atoi(num_blocks_str);

  create_team(num_blocks, 1024 * 512, &h_team, &d_team);
}

void fini_fib() { }

__device__ void fib_task(worker_t* worker, task_t* task)
{
  int64_t n = (int64_t)get_private(task, 0);
  int64_t* r = (int64_t*)get_private(task, 1);

  if (n < 2) {
    if (gpu_utils::thread_id() == 0) *r = n;
    return;
  }

  int64_t* r1 = (int64_t*)((void**)task->storage + task->num_privates + 0);
  int64_t* r2 = (int64_t*)((void**)task->storage + task->num_privates + 1);

  void* args1[2] = {(void*)(n - 1), r1};
  void* args2[2] = {(void*)(n - 2), r2};

  generate_task(worker, fib_task, 2, args1);
  generate_task(worker, fib_task, 2, args2);

  taskwait(worker);

  if (gpu_utils::thread_id() == 0) *r = *r1 + *r2;
}

double fib(int64_t n, int64_t* r)
{
  double t0 = omp_get_wtime();

  int64_t* h_result;
  CUDACHK(cudaGetSymbolAddress((void**)&h_result, d_result));

  const int nargs = 2;
  void* args[nargs] = {(void*)n, h_result};

  fork_team<fib_task>(h_team, d_team, nargs, args);
  CUDACHK(cudaMemcpy(r, h_result, sizeof(int64_t), cudaMemcpyDeviceToHost));
  reset_team(h_team, d_team);

  double t1 = omp_get_wtime();
  return t1 - t0;
}

