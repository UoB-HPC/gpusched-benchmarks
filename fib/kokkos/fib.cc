#include <omp.h>
#include <stdio.h>

#include <Kokkos_Core.hpp>

template <class Scheduler>
struct fib_task {
  using sched_type = Scheduler;
  using future_type = Kokkos::BasicFuture<int64_t, Scheduler>;
  using value_type = int64_t;

  int64_t n;
  future_type fib_m1;
  future_type fib_m2;

  KOKKOS_INLINE_FUNCTION
  fib_task(const int64_t arg_n) : fib_m1(), fib_m2(), n(arg_n) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(typename sched_type::member_type& member, value_type& result)
  {
    if (n < 2) {
      result = n;
    }
    else if (!fib_m1.is_null() && !fib_m2.is_null()) {
      result = fib_m1.get() + fib_m2.get();
    }
    else {
      auto& sched = member.scheduler();
      fib_m2 = Kokkos::task_spawn(
          Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High),
          fib_task(n - 2));
      fib_m1 = Kokkos::task_spawn(Kokkos::TaskSingle(sched), fib_task(n - 1));
      Kokkos::BasicFuture<void, Scheduler> dep[] = {fib_m1, fib_m2};
      Kokkos::BasicFuture<void, Scheduler> fib_all = sched.when_all(dep, 2);
      Kokkos::respawn(this, fib_all, Kokkos::TaskPriority::High);
    }
  }
};

void init_fib() { Kokkos::initialize(); }
void fini_fib() { Kokkos::finalize(); }

double fib(int64_t n, int64_t* r)
{
  double t0 = omp_get_wtime();
  using Scheduler =
      Kokkos::TaskSchedulerMultiple<Kokkos::DefaultExecutionSpace>;

  const size_t min_block_size = 32;
  const size_t max_block_size = 128;
  const size_t super_block_size = 10000;
  const size_t memory_capacity = 1024 * 1024 * 1024;

  Scheduler sched(typename Scheduler::memory_space(), memory_capacity,
                  min_block_size, std::min(max_block_size, memory_capacity),
                  std::min(super_block_size, memory_capacity));

  Kokkos::BasicFuture<int64_t, Scheduler> f =
      Kokkos::host_spawn(Kokkos::TaskSingle(sched), fib_task<Scheduler>(n));
  Kokkos::wait(sched);
  *r = f.get();

  double t1 = omp_get_wtime();
  return t1 - t0;
}
