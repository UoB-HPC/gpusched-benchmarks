#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "../common.hh"

int mycount = 0;
#pragma omp threadprivate(mycount)

int ok(int n, char* a)
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

void nqueens_ser(int n, int j, char* a)
{
  if (n == j) {
    mycount++;
    return;
  }

  for (int i = 0; i < n; ++i) {
    a[j] = (char)i;
    if (ok(j + 1, a)) {
      nqueens_ser(n, j + 1, a);
    }
  }
}

void nqueens(int n, int j, char* a, int depth)
{
  if (n == j) {
    mycount++;
    return;
  }

  for (int i = 0; i < n; ++i) {
    if (depth < CUTOFF) {
#pragma omp task
      {
        char b[MAX_SOLUTIONS];
        memcpy(b, a, j * sizeof(char));
        b[j] = (char)i;
        if (ok(j + 1, b)) {
          nqueens(n, j + 1, b, depth + 1);
        }
      }
    }
    else {
      a[j] = (char)i;
      if (ok(j + 1, a)) {
        nqueens_ser(n, j + 1, a);
      }
    }
  }
#pragma omp taskwait
}

double find_nqueens(int size, int* total_count)
{
  double t0 = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single
    {
      char a[MAX_SOLUTIONS];
      nqueens(size, 0, a, 0);
    }
#pragma omp atomic
    *total_count += mycount;
  }
  double t1 = omp_get_wtime();

  return (t1 - t0);
}
