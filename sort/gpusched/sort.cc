#include <gpusched.h>
#include <omp.h>
#include <stdio.h>

#define CUTOFF 2 * 1024
#define CUTOFF1 2 * 1024
#define CUTOFF2 20

#define ELM long

static team_t* h_team = nullptr;
static team_t* d_team = nullptr;
static __device__ team_t* device_d_team = nullptr;

__device__ static worker_t* get_worker()
{
  return device_d_team->workers + gpu_utils::global_worker_id();
}

__device__ static void warp_memcpy(long* dst, long* src, size_t n)
{
  for (int i = gpu_utils::thread_id(); i < n; i += gpu_utils::warp_size) {
    dst[i] = src[i];
  }
}

__device__ static inline ELM med3(ELM a, ELM b, ELM c)
{
  if (a < b) {
    if (b < c) {
      return b;
    }
    else {
      if (a < c)
        return c;
      else
        return a;
    }
  }
  else {
    if (b > c) {
      return b;
    }
    else {
      if (a > c)
        return c;
      else
        return a;
    }
  }
}

/*
 * simple approach for now; a better median-finding
 * may be preferable
 */
__device__ static inline ELM choose_pivot(ELM* low, ELM* high)
{
  return med3(*low, *high, low[(high - low) / 2]);
}

__device__ static ELM* seqpart(ELM* low, ELM* high)
{
  ELM pivot;
  ELM h, l;
  ELM* curr_low = low;
  ELM* curr_high = high;

  pivot = choose_pivot(low, high);

  while (1) {
    while ((h = *curr_high) > pivot) curr_high--;

    while ((l = *curr_low) < pivot) curr_low++;

    if (curr_low >= curr_high) break;

    *curr_high-- = l;
    *curr_low++ = h;
  }

  /*
   * I don't know if this is really necessary.
   * The problem is that the pivot is not always the
   * first element, and the partition may be trivial.
   * However, if the partition is trivial, then
   * *high is the largest element, whence the following
   * code.
   */
  if (curr_high < high)
    return curr_high;
  else
    return curr_high - 1;
}

#define swap(a, b)                                                             \
  {                                                                            \
    ELM tmp;                                                                   \
    tmp = a;                                                                   \
    a = b;                                                                     \
    b = tmp;                                                                   \
  }

__device__ static void insertion_sort(ELM* low, ELM* high)
{
  ELM *p, *q;
  ELM a, b;

  for (q = low + 1; q <= high; ++q) {
    a = q[0];
    for (p = q - 1; p >= low && (b = p[0]) > a; p--) p[1] = b;
    p[1] = a;
  }
}

/*
 * tail-recursive quicksort, almost unrecognizable :-)
 */
__device__ void seqquick(ELM* low, ELM* high)
{
  if (gpu_utils::thread_id() == 0) {
    ELM* p;

    while (high - low >= CUTOFF2) {
      p = seqpart(low, high);
      seqquick(low, p);
      low = p + 1;
    }

    insertion_sort(low, high);
  }
}

__device__ void seqmerge(ELM* low1, ELM* high1, ELM* low2, ELM* high2,
                         ELM* lowdest)
{
  ELM a1, a2;

  /*
   * The following 'if' statement is not necessary
   * for the correctness of the algorithm, and is
   * in fact subsumed by the rest of the function.
   * However, it is a few percent faster.  Here is why.
   *
   * The merging loop below has something like
   *   if (a1 < a2) {
   *        *dest++ = a1;
   *        ++low1;
   *        if (end of array) break;
   *        a1 = *low1;
   *   }
   *
   * Now, a1 is needed immediately in the next iteration
   * and there is no way to mask the latency of the load.
   * A better approach is to load a1 *before* the end-of-array
   * check; the problem is that we may be speculatively
   * loading an element out of range.  While this is
   * probably not a problem in practice, yet I don't feel
   * comfortable with an incorrect algorithm.  Therefore,
   * I use the 'fast' loop on the array (except for the last
   * element) and the 'slow' loop for the rest, saving both
   * performance and correctness.
   */

  // No obvious gain from SMT
  if (gpu_utils::thread_id() == 0) {
    if (low1 < high1 && low2 < high2) {
      a1 = *low1;
      a2 = *low2;
      for (;;) {
        if (a1 < a2) {
          *lowdest++ = a1;
          a1 = *++low1;
          if (low1 >= high1) break;
        }
        else {
          *lowdest++ = a2;
          a2 = *++low2;
          if (low2 >= high2) break;
        }
      }
    }
    if (low1 <= high1 && low2 <= high2) {
      a1 = *low1;
      a2 = *low2;
      for (;;) {
        if (a1 < a2) {
          *lowdest++ = a1;
          ++low1;
          if (low1 > high1) break;
          a1 = *low1;
        }
        else {
          *lowdest++ = a2;
          ++low2;
          if (low2 > high2) break;
          a2 = *low2;
        }
      }
    }
  }

  // values not present at all threads, broadcast
  // high values not modified
  low1 = (long*)__shfl_sync(0xffffffff, (uint64_t)low1, 0, 32);
  low2 = (long*)__shfl_sync(0xffffffff, (uint64_t)low2, 0, 32);
  lowdest = (long*)__shfl_sync(0xffffffff, (uint64_t)lowdest, 0, 32);

  if (low1 > high1) {
    warp_memcpy(lowdest, low2, (high2 - low2 + 1));
  }
  else {
    warp_memcpy(lowdest, low1, (high1 - low1 + 1));
  }
}

#define swap_indices(a, b)                                                     \
  {                                                                            \
    ELM* tmp;                                                                  \
    tmp = a;                                                                   \
    a = b;                                                                     \
    b = tmp;                                                                   \
  }

__device__ ELM* binsplit(ELM val, ELM* low, ELM* high)
{
  /*
   * returns index which contains greatest element <= val.  If val is
   * less than all elements, returns low-1
   */
  ELM* mid;

  while (low != high) {
    mid = low + ((high - low + 1) >> 1);
    if (val <= *mid)
      high = mid - 1;
    else
      low = mid;
  }

  if (*low > val)
    return low - 1;
  else
    return low;
}

__device__ void cilkmerge_par(ELM* low1, ELM* high1, ELM* low2, ELM* high2,
                              ELM* lowdest);

__device__ void cilkmerge_par_task(worker_t* worker, task_t* task)
{
  ELM* low1 = (ELM*)get_private(task, 0);
  ELM* high1 = (ELM*)get_private(task, 1);
  ELM* low2 = (ELM*)get_private(task, 2);
  ELM* high2 = (ELM*)get_private(task, 3);
  ELM* lowdest = (ELM*)get_private(task, 4);

  cilkmerge_par(low1, high1, low2, high2, lowdest);
}

__device__ void cilkmerge_par(ELM* low1, ELM* high1, ELM* low2, ELM* high2,
                              ELM* lowdest)
{
  /*
   * Cilkmerge: Merges range [low1, high1] with range [low2, high2]
   * into the range [lowdest, ...]
   */

  ELM *split1, *split2; /*
                         * where each of the ranges are broken for
                         * recursive merge
                         */
  long int lowsize;     /*
                         * total size of lower halves of two
                         * ranges - 2
                         */

  /*
   * We want to take the middle element (indexed by split1) from the
   * larger of the two arrays.  The following code assumes that split1
   * is taken from range [low1, high1].  So if [low1, high1] is
   * actually the smaller range, we should swap it with [low2, high2]
   */

  // Only swapping local ptrs, all threads should be able to participate
  if (high2 - low2 > high1 - low1) {
    swap_indices(low1, low2);
    swap_indices(high1, high2);
  }

  // Let all threads help with memcpy
  if (high2 < low2) {
    /* smaller range is empty */
    warp_memcpy(lowdest, low1, (high1 - low1));
    return;
  }

  // Let all threads help with seqmerge
  if (high2 - low2 < CUTOFF) {
    seqmerge(low1, high1, low2, high2, lowdest);
    return;
  }
  /*
   * Basic approach: Find the middle element of one range (indexed by
   * split1). Find where this element would fit in the other range
   * (indexed by split 2). Then merge the two lower halves and the two
   * upper halves.
   */

  split1 = ((high1 - low1 + 1) / 2) + low1;
  if (gpu_utils::thread_id() == 0) split2 = binsplit(*split1, low2, high2);
  split2 = (long*)__shfl_sync(0xffffffff, (uint64_t)split2, 0, 32);
  lowsize = split1 - low1 + split2 - low2;

  if (gpu_utils::thread_id() == 0) *(lowdest + lowsize + 1) = *split1;

  long* args1[5] = {low1, split1 - 1, low2, split2, lowdest};
  long* args2[5] = {split1 + 1, high1, split2 + 1, high2,
                    lowdest + lowsize + 2};

  worker_t* worker = get_worker();
  generate_task(worker, cilkmerge_par_task, 5, (void**)args1);
  generate_task(worker, cilkmerge_par_task, 5, (void**)args2);
  taskwait(worker);

  //#pragma omp task untied
  // cilkmerge_par(low1, split1 - 1, low2, split2, lowdest);
  //#pragma omp task untied
  // cilkmerge_par(split1 + 1, high1, split2 + 1, high2, lowdest + lowsize + 2);
  //#pragma omp taskwait

  return;
}
__device__ void cilksort_par(ELM* low, ELM* tmp, long size);

__device__ void cilksort_par_task(worker_t* worker, task_t* task)
{
  ELM* low = (ELM*)get_private(task, 0);
  ELM* tmp = (ELM*)get_private(task, 1);
  long size = (long)get_private(task, 2);

  cilksort_par(low, tmp, size);
}

__device__ void cilksort_par(ELM* low, ELM* tmp, long size)
{
  /*
   * divide the input in four parts of the same size (A, B, C, D)
   * Then:
   *   1) recursively sort A, B, C, and D (in parallel)
   *   2) merge A and B into tmp1, and C and D into tmp2 (in parallel)
   *   3) merge tmp1 and tmp2 into the original array
   */
  long quarter = size / 4;
  ELM *A, *B, *C, *D, *tmpA, *tmpB, *tmpC, *tmpD;

  // No easy gains for vectorisation here
  if (size < CUTOFF1) {
    /* quicksort when less than 1024 elements */
    seqquick(low, low + size - 1);
    return;
  }

  A = low;
  tmpA = tmp;
  B = A + quarter;
  tmpB = tmpA + quarter;
  C = B + quarter;
  tmpC = tmpB + quarter;
  D = C + quarter;
  tmpD = tmpC + quarter;

  ////#pragma omp task untied
  // cilksort_par(A, tmpA, quarter);
  ////#pragma omp task untied
  // cilksort_par(B, tmpB, quarter);
  ////#pragma omp task untied
  // cilksort_par(C, tmpC, quarter);
  ////#pragma omp task untied
  // cilksort_par(D, tmpD, size - 3 * quarter);
  ////#pragma omp taskwait

  // use same args array to save potentially save on stack space
  worker_t* worker = get_worker();
  void* args[5] = {(void*)A, (void*)tmpA, (void*)quarter};
  generate_task(worker, cilksort_par_task, 3, args);

  args[0] = (void*)B;
  args[1] = (void*)tmpB;
  generate_task(worker, cilksort_par_task, 3, args);

  args[0] = (void*)C;
  args[1] = (void*)tmpC;
  generate_task(worker, cilksort_par_task, 3, args);

  args[0] = (void*)D;
  args[1] = (void*)tmpD;
  args[2] = (void*)(size - 3 * quarter);
  generate_task(worker, cilksort_par_task, 3, args);

  taskwait(worker);

  //#pragma omp task untied
  // cilkmerge_par(A, A + quarter - 1, B, B + quarter - 1, tmpA);
  //#pragma omp task untied
  // cilkmerge_par(C, C + quarter - 1, D, low + size - 1, tmpC);
  //#pragma omp taskwait

  args[0] = (void*)A;
  args[1] = (void*)(A + quarter - 1);
  args[2] = (void*)B;
  args[3] = (void*)(B + quarter - 1);
  args[4] = (void*)(tmpA);
  generate_task(worker, cilkmerge_par_task, 5, args);

  args[0] = (void*)C;
  args[1] = (void*)(C + quarter - 1);
  args[2] = (void*)D;
  args[3] = (void*)(low + size - 1);
  args[4] = (void*)(tmpC);
  generate_task(worker, cilkmerge_par_task, 5, args);

  taskwait(worker);

  cilkmerge_par(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A);
}

__device__ void entry_point(worker_t* worker, task_t* task)
{
  long* array = (long*)get_private(task, 0);
  long* tmp = (long*)get_private(task, 1);
  long n = (long)get_private(task, 2);

  device_d_team = worker->team;

  cilksort_par(array, tmp, n);
}

double sort_par(long* array, long* tmp, long n)
{
  long* d_array;
  long* d_tmp;
  CUDACHK(cudaMalloc(&d_array, sizeof(long) * n));
  CUDACHK(cudaMalloc(&d_tmp, sizeof(long) * n));

  CUDACHK(cudaMemcpy(d_array, array, sizeof(long) * n, cudaMemcpyHostToDevice));

  const char* num_blocks_str = getenv("GPUSCHED_NUM_BLOCKS");
  const int num_blocks =
      (num_blocks_str == NULL) ? 56 * 5 : atoi(num_blocks_str);

  create_team(num_blocks, 1024 * 512, &h_team, &d_team);

  const int nargs = 3;
  void* args[nargs] = {(void*)d_array, (void*)d_tmp, (void*)n};

  double t0 = omp_get_wtime();
  fork_team<entry_point>(h_team, d_team, nargs, args);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());
  double t1 = omp_get_wtime();

  CUDACHK(cudaMemcpy(array, d_array, sizeof(long) * n, cudaMemcpyDeviceToHost));

  return t1 - t0;
}
