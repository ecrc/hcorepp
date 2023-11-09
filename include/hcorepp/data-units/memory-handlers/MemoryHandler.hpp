/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_DATA_UNITS_MEMORY_HANDLER_HPP
#define HCOREPP_DATA_UNITS_MEMORY_HANDLER_HPP

#include <functional>
#include <iostream>
#include <cstddef>
#include "hcorepp/common/Definitions.hpp"
#include "hcorepp/kernels/RunContext.hpp"
#include "hcorepp/helpers/DebuggingTimer.hpp"
#include <unordered_set>

#ifdef USE_CUDA
#include "pool/MemoryHandler.hpp"
#elif defined(USE_SYCL)

#include "pool/MemoryHandler.hpp"

#else
#include "on-demand/MemoryHandler.hpp"
#endif


#endif //HCOREPP_DATA_UNITS_MEMORY_HANDLER_HPP

