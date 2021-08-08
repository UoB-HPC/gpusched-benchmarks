#include <stdio.h>
#include <stdlib.h>

#include <unordered_map>

unsigned long rand_nxt = 0;

static inline unsigned long my_rand(void)
{
  rand_nxt = rand_nxt * 1103515245 + 12345;
  return rand_nxt;
}

static inline void my_srand(unsigned long seed) { rand_nxt = seed; }

void fill_array(long* array, int n)
{
  for (int i = 0; i < n; ++i) {
    array[i] = i;
  }
}

void scramble_array(long* array, int n)
{
  my_srand(1);
  for (int i = 0; i < n; ++i) {
    unsigned long j = my_rand() % n;
    // swap
    long tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
  }
}

void sort_init(long* array, int n)
{
  fill_array(array, n);
  scramble_array(array, n);
}

int sort_verify(long* array, int n)
{
  int success = 1;
  for (int i = 0; i < n; ++i) {
    if (array[i] != i) success = 0;
  }
  return success;
}

static std::unordered_map<long, long> map = {
    {32 * 1024 * 1024, 404922},
    {64 * 1024 * 1024, 701568},
    {128 * 1024 * 1024, 1797922},
    {256 * 1024 * 1024, 3162412},
};

long get_num_tasks(long n)
{
  auto it = map.find(n);
  if (it != map.end()) {
    return it->second;
  }
  else
    return -1;
}

extern double sort_par(long* array, long* tmp, long n);

int main(int argc, char** argv)
{
  const long n = (argc >= 2) ? atol(argv[1]) : 32 * 1024 * 1024;

  if (n < 2 * 1024) {
    printf("array has to be bigger than 2048, n = %ld\n", n);
    exit(1);
  }

  printf("n = %ld\n", n);

  long* array = (long*)malloc(sizeof(long) * n);
  long* tmp = (long*)malloc(sizeof(long) * n);

  printf("size = %f MB\n", ((double)n * sizeof(long)) / 1e6);

  sort_init(array, n);

  double time = sort_par(array, tmp, n);
  printf("time = %f\n", time);

  int success = sort_verify(array, n);
  const char* success_str = (success) ? "pass" : "fail";

  long num_tasks = get_num_tasks(n);
  char num_tasks_str[128];
  char task_rate_str[128];
  if (num_tasks == -1) {
    sprintf(num_tasks_str, "%s", "-");
    sprintf(task_rate_str, "%s", "-");
  }
  else {
    sprintf(num_tasks_str, "%ld", num_tasks);
    sprintf(task_rate_str, "%e", ((double)num_tasks)/time);
  }

  printf("Benchmark,n,time,num tasks,task rate,success\n");
  printf("SORT,%ld,%f,%s,%s,%s\n", n, time, num_tasks_str, task_rate_str,
         success_str);
}
