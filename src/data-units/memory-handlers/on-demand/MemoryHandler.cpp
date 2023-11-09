/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */
#include <hcorepp/data-units/memory-handlers/on-demand/MemoryHandler.hpp>
#include <hcorepp/kernels/memory.hpp>

namespace hcorepp {
    namespace dataunits {

        template <typename T>
        MemoryHandler<T>* MemoryHandler<T>::mpInstance = nullptr;

        template<typename T>
        MemoryHandler<T>::MemoryHandler()
                : mContextManager(kernels::ContextManager::GetInstance()) {
            mUnits.emplace_back(mContextManager.GetContext(0));
            for(size_t i = 1; i < mContextManager.GetNumOfContexts(); i++) {
                mUnits.emplace_back(mContextManager.GetContext(i));
            }
        }

        template<typename T>
        void MemoryHandler<T>::Print(std::ostream &aOutStream) const {
            aOutStream << "On Demand Handler has : " << mUnits.size() << " units currently";
        }

        template<typename T>
        T *MemoryHandler<T>::RequestAllocation(size_t aSize, size_t aIdx) {
            return mUnits[aIdx].RequestAllocation(aSize);
        }

        template<typename T>
        void MemoryHandler<T>::BufferMemSet(char aValue, size_t aSize, size_t aOffset, size_t aIdx) {
            return mUnits[aIdx].BufferMemSet(aValue, aSize, aOffset);
        }

        template<typename T>
        void MemoryHandler<T>::FreeAllocations() {
            for(auto& unit: mUnits) {
                unit.FreeAllocations();
            }
        }

        template<typename T>
        size_t MemoryHandler<T>::GetPoolSize() const {
            return 0;
        }

        template<typename T>
        MemoryHandler<T>& MemoryHandler<T>::GetInstance() {
            if(MemoryHandler<T>::mpInstance == nullptr) {
                MemoryHandler<T>::mpInstance = new MemoryHandler<T>();
            }
            return *MemoryHandler<T>::mpInstance;
        }

        HCOREPP_INSTANTIATE_CLASS(MemoryHandler)
        HCOREPP_INSTANTIATE_CLASS(MemoryUnit)

    }
}