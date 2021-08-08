#include <stdio.h>
#include <stdlib.h>

extern double fib(int64_t n, int64_t* r);
extern void init_fib();
extern void fini_fib();

int64_t get_num_tasks(int64_t n)
{
  const int64_t mask = 0x03;
  int64_t count[4] = {1, 1, 0, 0};
  for (int64_t i = 2; i <= n; ++i) {
    count[i & mask] = 1 + count[(i - 1) & mask] + count[(i - 2) & mask];
  }
  return count[n & mask];
}

int64_t get_solution(int64_t n)
{
  const int64_t mask = 0x03;
  int64_t fib[4] = {0, 1, 1, 2};
  for (int64_t i = 2; i <= n; ++i) {
    fib[i & mask] = fib[(i - 1) & mask] + fib[(i - 2) & mask];
  }
  return fib[n & mask];
}

int main(int argc, char** argv)
{
  int n = (argc < 2) ? 20 : atoi(argv[1]);
  int nreps = (argc < 3) ? 5 : atoi(argv[2]);

  int64_t num_tasks = get_num_tasks(n);
  int64_t solution = get_solution(n);
  bool pass = true;

  init_fib();

  double avg_time = 0.0;
  for (int i = 0; i < nreps; ++i) {
    int64_t r;
    double t = fib(n, &r);

    if (r != solution) pass = false;
    avg_time += t;
#ifdef DEBUG
    printf("  rep %2d time = %8.5f, result = %lld\n", i, t, r);
#endif
  }
  avg_time /= (double)nreps;

  fini_fib();

  const double task_rate = (double)num_tasks / avg_time;
  printf("-FIBRESULT-%2d,%4d,%12.8f,%12.5e,%12ld,%12ld,%4s\n", n, nreps, avg_time,
         task_rate, num_tasks, solution, (pass) ? "pass" : "fail");

  // printf("pass = %s\n", (pass) ? "pass" : "fail");
  // printf("solution = %ld\n", solution);
  // printf("average time = %f\n", avg_time);
  // printf("num tasks = %ld\n", num_tasks);
  // printf("tasks/s = %e\n", (double)num_tasks / avg_time);
}
