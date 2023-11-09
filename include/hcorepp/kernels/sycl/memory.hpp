/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_SYCL_MEMORY_H
#define HCOREPP_KERNELS_SYCL_MEMORY_H

#include <cstring>

namespace hcorepp {
    namespace memory {

        template<typename T>
        T *AllocateArray(size_t aNumElements, const hcorepp::kernels::RunContext &aContext) {
            T *array = (T *) sycl::malloc_device((aNumElements * sizeof(T)), aContext.GetQueue());
            aContext.AddVariable(array, aNumElements * sizeof(T));
            return array;
        }

        template<typename T>
        void DestroyArray(T *apArray, const hcorepp::kernels::RunContext &aContext) {
            aContext.RemoveVariable(apArray);
            sycl::free(apArray, aContext.GetQueue());
        }


        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, size_t aNumOfElements,
                    const hcorepp::kernels::RunContext &aContext, MemoryTransfer aTransferType, bool aBlocking) {
            std::vector<kernels::DepPair> actions(
                    {kernels::DepPair((void *) apDestination, kernels::VariableDependency::WRITE),
                     kernels::DepPair((void *) apSrcDataArray, kernels::VariableDependency::READ)});
            auto deps = aContext.GetVariableEvents(actions);
            auto ev = aContext.GetQueue().submit([&](sycl::handler &h) {
                h.depends_on(deps);
                h.memcpy(apDestination, apSrcDataArray, aNumOfElements * sizeof(T));
            });
            aContext.SetVariableEvent(actions, ev);
            if (aBlocking) {
                ev.wait();
            }
        }

        template<typename T>
        void Memset(T *apDestination, char aValue, size_t aNumOfElements,
                    const hcorepp::kernels::RunContext &aContext) {
            auto deps = aContext.GetVariableEvents({{apDestination, kernels::VariableDependency::WRITE}});
            auto ev = aContext.GetQueue().submit([&](sycl::handler &h) {
                h.depends_on(deps);
                h.memset(apDestination, aValue, aNumOfElements * sizeof(T));
            });
            aContext.SetVariableEvent({{apDestination, kernels::VariableDependency::WRITE}}, ev);
        }
    }//namespace memory
}//namespace hcorepp

#endif //HCOREPP_KERNELS_SYCL_MEMORY_H
