#include <omp.h>
#include <cstring>

#define CUTOFF 2 * 1024
#define CUTOFF1 2 * 1024
#define CUTOFF2 20

#define ELM long

static inline ELM med3(ELM a, ELM b, ELM c)
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
static inline ELM choose_pivot(ELM* low, ELM* high)
{
  return med3(*low, *high, low[(high - low) / 2]);
}

static ELM* seqpart(ELM* low, ELM* high)
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

static void insertion_sort(ELM* low, ELM* high)
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
void seqquick(ELM* low, ELM* high)
{
  ELM* p;

  while (high - low >= CUTOFF2) {
    p = seqpart(low, high);
    seqquick(low, p);
    low = p + 1;
  }

  insertion_sort(low, high);
}

void seqmerge(ELM* low1, ELM* high1, ELM* low2, ELM* high2, ELM* lowdest)
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
  if (low1 > high1) {
    memcpy(lowdest, low2, sizeof(ELM) * (high2 - low2 + 1));
  }
  else {
    memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1 + 1));
  }
}

#define swap_indices(a, b)                                                     \
  {                                                                            \
    ELM* tmp;                                                                  \
    tmp = a;                                                                   \
    a = b;                                                                     \
    b = tmp;                                                                   \
  }

ELM* binsplit(ELM val, ELM* low, ELM* high)
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

void cilkmerge_par(ELM* low1, ELM* high1, ELM* low2, ELM* high2, ELM* lowdest)
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

  if (high2 - low2 > high1 - low1) {
    swap_indices(low1, low2);
    swap_indices(high1, high2);
  }
  if (high2 < low2) {
    /* smaller range is empty */
    memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1));
    return;
  }
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
  split2 = binsplit(*split1, low2, high2);
  lowsize = split1 - low1 + split2 - low2;

  /*
   * directly put the splitting element into
   * the appropriate location
   */
  *(lowdest + lowsize + 1) = *split1;
#pragma omp task untied
  cilkmerge_par(low1, split1 - 1, low2, split2, lowdest);
#pragma omp task untied
  cilkmerge_par(split1 + 1, high1, split2 + 1, high2, lowdest + lowsize + 2);
#pragma omp taskwait

  return;
}

void cilksort_par(ELM* low, ELM* tmp, long size)
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

#pragma omp task untied
  cilksort_par(A, tmpA, quarter);
#pragma omp task untied
  cilksort_par(B, tmpB, quarter);
#pragma omp task untied
  cilksort_par(C, tmpC, quarter);
#pragma omp task untied
  cilksort_par(D, tmpD, size - 3 * quarter);
#pragma omp taskwait

#pragma omp task untied
  cilkmerge_par(A, A + quarter - 1, B, B + quarter - 1, tmpA);
#pragma omp task untied
  cilkmerge_par(C, C + quarter - 1, D, low + size - 1, tmpC);
#pragma omp taskwait

  cilkmerge_par(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A);
}

double sort_par(long* array, long* tmp, long n)
{
  double t0 = omp_get_wtime();
#pragma omp parallel
#pragma omp single
#pragma omp task untied
  cilksort_par(array, tmp, n);
  double t1 = omp_get_wtime();
  return t1 - t0;

}

