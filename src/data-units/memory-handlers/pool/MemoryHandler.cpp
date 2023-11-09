/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */
#include <hcorepp/data-units/memory-handlers/pool/MemoryHandler.hpp>
#include <hcorepp/kernels/memory.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        MemoryUnit<T>::MemoryUnit(const hcorepp::kernels::RunContext& aContext, size_t aAllocationSize) : mSize(aAllocationSize), mAvailable(aAllocationSize), mNext(0), mContext(aContext)
        {
            if (aAllocationSize > 0) {
                mpDataArray = hcorepp::memory::AllocateArray<T>(aAllocationSize, aContext);
                hcorepp::memory::Memset(mpDataArray, 0, mSize, aContext);
            } else {
                mpDataArray = nullptr;
            }
        }

        template<typename T>
        void MemoryUnit<T>::Initialize(size_t aSize) {
            if (IsInitialized()) {
                throw std::runtime_error("MemoryUnit Is Already Allocated");
            }
            if (aSize > 0) {
                mpDataArray = hcorepp::memory::AllocateArray<T>(aSize, mContext);
                hcorepp::memory::Memset(mpDataArray, 0, aSize, mContext);
                mAvailable = aSize;
                mSize = aSize;
                mNext = 0;
            }
        }

        template<typename T>
        void MemoryUnit<T>::Free() {
            if (mpDataArray) {
                hcorepp::memory::DestroyArray(mpDataArray, mContext);
                mSize = 0;
                mNext = 0;
                mAvailable = 0;
                mpDataArray = nullptr;
            }
        }

        template<typename T>
        void MemoryUnit<T>::BufferMemSet(char aValue, size_t aSize, size_t aOffset) {
            if (mpDataArray) {
                hcorepp::memory::Memset(&(mpDataArray[aOffset]), aValue, aSize, mContext);
            }
        }

        template <typename T>
        MemoryHandler<T>* MemoryHandler<T>::mpInstance = nullptr;

        template<typename T>
        MemoryHandler<T>::MemoryHandler(const size_t aAllocationSize) : mContextManager(kernels::ContextManager::GetInstance()) {
            mPools.emplace_back(mContextManager.GetContext(0), aAllocationSize);
            for(size_t i = 1; i < mContextManager.GetNumOfContexts(); i++) {
                        mPools.emplace_back(mContextManager.GetContext(i), 0);
            }
        }

        template<typename T>
        MemoryHandler<T> &MemoryHandler<T>::GetInstance() {
            if(MemoryHandler<T>::mpInstance == nullptr) {
                MemoryHandler<T>::mpInstance = new MemoryHandler<T>();
            }
            return *MemoryHandler<T>::mpInstance;
        }

        template<typename T>
        MemoryUnit<T>& MemoryHandler<T>::GetMemoryUnit(size_t aIdx) {
            return mPools[aIdx];
        }

        template<typename T>
        size_t MemoryHandler<T>::GetPoolSize(size_t aIdx) const {
            return mPools[aIdx].GetPoolSize();
        }

        template<typename T>
        size_t MemoryHandler<T>::GetPoolAvailableSize(size_t aIdx) const {
            return mPools[aIdx].GetPoolAvailableSize();
        }

        template<typename T>
        bool MemoryHandler<T>::IsInitialized(size_t aIdx) const {
            return mPools[aIdx].IsInitialized();
        }

        template<typename T>
        T *MemoryHandler<T>::RequestAllocation(size_t aSize, size_t aIdx) {
            return mPools[aIdx].RequestAllocation(aSize);
        }

        template<typename T>
        void MemoryHandler<T>::Initialize(size_t aSize, size_t aIdx) {
            mPools[aIdx].Initialize(aSize);
        }

        template<typename T>
        void MemoryHandler<T>::BufferMemSet(char aValue, size_t aSize, size_t aOffset, size_t aIdx) {
            mPools[aIdx].BufferMemSet(aValue, aSize, aOffset);
        }

        template<typename T>
        void MemoryHandler<T>::FreeAllocations() {
            for(auto& pool : mPools) {
                pool.Free();
            }
        }

        template<typename T>
        void MemoryHandler<T>::FreePool(size_t aIdx) {
            mPools[aIdx].Free();
        }

        HCOREPP_INSTANTIATE_CLASS(MemoryHandler)
        HCOREPP_INSTANTIATE_CLASS(MemoryUnit)

    }
}