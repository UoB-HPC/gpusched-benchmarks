#include <stdio.h>
#include <stdint.h> 
#include <omp.h>

void init_fib() { }
void fini_fib() { }

int64_t fib_task(int64_t n)
{
  if (n < 2) return n;
  int64_t r1, r2;

#pragma omp task shared(r2) 
  r2 = fib_task(n - 2);

#pragma omp task shared(r1) 
  r1 = fib_task(n - 1);

#pragma omp taskwait

  return r1 + r2;
}

double fib(int64_t n, int64_t* r)
{
  double t0 = omp_get_wtime();
#pragma omp parallel 
#pragma omp single
  *r = fib_task(n);

  double t1 = omp_get_wtime();

  return t1 - t0;
}
