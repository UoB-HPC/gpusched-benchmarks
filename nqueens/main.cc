#include <stdio.h>
#include <stdlib.h>

#include "common.hh"

#define RESULT_SUCCESS 0
#define RESULT_FAIL 1
#define RESULT_NA 2
static const char* result_strings[] = {"success", "fail", "N/A"};

extern double find_nqueens(int size, int* total_count);

int verify_queens(int size, int total_count)
{
  printf("res = %d, ans = %d\n", total_count, solutions[size - 1]);
  if (size > MAX_SOLUTIONS) return -1;
  if (total_count == solutions[size - 1]) return RESULT_SUCCESS;
  return RESULT_FAIL;
}

int main(int argc, char** argv)
{
  const int size = (argc >= 2) ? atoi(argv[1]) : MAX_SOLUTIONS;

  int total_count = 0;
  double time = find_nqueens(size, &total_count);
  printf("time = %f\n", time);

  int res = verify_queens(size, total_count);

  printf("result = %s\n", result_strings[res]);
}

