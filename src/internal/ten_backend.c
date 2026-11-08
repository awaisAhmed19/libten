#include "internal/ten_backend.h"
#include "backend/cpu/ten_memory.h"

#ifdef USE_CUDA
#include "backend/cuda/ten_memory.h"
#endif

static const Ten_Backend cpu_backend = {.memory = {.allocate = cpu_allocate,
                                                   .deallocate = cpu_deallocate,
                                                   .copy = cpu_copy}};

#ifdef USE_CUDA
static const Ten_Backend cuda_backend = {
    .memory = {.allocate = cuda_allocate,
               .deallocate = cuda_deallocate,
               .copy = cuda_copy}};
#endif

const Ten_Backend *T_get_backend(Device device) {
  switch (device.devtype) {
  case CPU:
    return &cpu_backend;

  case CUDA:
#ifdef USE_CUDA
    return &cuda_backend;
#else
    return NULL;
#endif

  default:
    return NULL;
  }
}
