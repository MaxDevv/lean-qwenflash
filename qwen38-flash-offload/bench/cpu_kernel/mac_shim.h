// macOS/Rosetta build shim for correctness runs only (no pinning, no THP, no perf counters)
#pragma once
#ifdef __APPLE__
#include <pthread.h>
#include <sys/mman.h>
typedef struct { int x; } cpu_set_t;
#define CPU_ZERO(s) ((void)(s))
#define CPU_SET(c, s) ((void)(c), (void)(s))
static inline int pthread_setaffinity_np(pthread_t, size_t, const cpu_set_t*) { return 0; }
static inline int sched_getaffinity(int, size_t, cpu_set_t*) { return 0; }
static inline int sched_setaffinity(int, size_t, const cpu_set_t*) { return 0; }
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
#endif
