#ifndef HCOREPP_KERNELS_CUDA_ERROR_CHECKING_H
#define HCOREPP_KERNELS_CUDA_ERROR_CHECKING_H

#include <cuda_runtime.h>
#include <cstdio>

namespace hcorepp {

    #define GPU_ERROR_CHECK(ans) { hcorepp::AssertGPU((ans), __FILE__, __LINE__); }

    inline void AssertGPU(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
}

#endif //HCOREPP_KERNELS_CUDA_ERROR_CHECKING_H
