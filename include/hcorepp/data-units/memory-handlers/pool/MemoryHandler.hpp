/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_DATA_UNITS_POOL_MEMORY_HANDLER_HPP
#define HCOREPP_DATA_UNITS_POOL_MEMORY_HANDLER_HPP

#include <functional>
#include <iostream>
#include <cstddef>
#include "hcorepp/common/Definitions.hpp"
#include "hcorepp/kernels/RunContext.hpp"
#include "hcorepp/helpers/DebuggingTimer.hpp"
#include <unordered_set>
#include <hcorepp/kernels/ContextManager.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        class MemoryUnit {
            /** Pointer to Workspace/pool array */
            T* mpDataArray = nullptr;
            /** Size of MemoryUnit in number of elements */
            size_t mSize = 0;
            /** Available Size of MemoryUnit in number of elements  */
            size_t mAvailable = 0;
            /** Index of next available memory address(element not byte number) to be used */
            size_t mNext = 0;
            /** Context to be used for Memory Management functions for a specific pool object */
            const hcorepp::kernels::RunContext& mContext;
        public:
            explicit MemoryUnit(const hcorepp::kernels::RunContext& aContext, size_t aAllocationSize = 0);

            ~MemoryUnit() {
                this->FreeAllocations();
            };
            /**
             * @brief
             * Request allocation of a specific size from a specific pool
             *
             * @param[in] aSize
             * Memory Request Size
             *
             * @return
             * Pointer to the data array.
             *
             */
            [[nodiscard]] T *
            RequestAllocation(size_t aSize) {
                if (mAvailable < aSize || mpDataArray == nullptr) {
                    throw std::runtime_error("Requested Allocation Greater than MemoryUnit Size or the pool isn't initialized");
                }
                auto *ret = &(mpDataArray[mNext]);
                mNext += aSize;
                mAvailable -= aSize;
                return ret;
            }

            /**
             * @brief
             * The pool array getter.
             *
             * @return
             * Pointer to the pool.
             */
            T *
            GetData() const {
                return mpDataArray;
            }

            T *
            GetData()  {
                return mpDataArray;
            }

            [[nodiscard]] size_t
            GetPoolSize() const {
                return mSize;
            }

            /**
             * @brief
             * Available MemoryUnit Size getter.
             *
             * @return
             * Available MemoryUnit Size.
             */
            [[nodiscard]] size_t
            GetPoolAvailableSize() const {
                return mAvailable;
            }

            /**
             * @brief
             * If MemoryUnit Allocated, return true.
             *
             * @return
             */
            [[nodiscard]] bool
            IsInitialized() const {
                return mpDataArray != nullptr;
            }

            /**
             * @brief
             * Allocate MemoryUnit if not allocated.
             *
             * @param[in] aSize
             * Size of pool to be allocated
             */
            void
            Initialize(size_t aSize);

            /**
             * @brief
             * Set Memory MemoryUnit Value.
             * @param[in] aValue
             * Value to be set
             * @param[in] aSize
             * Size of pool to be set to specified value
             * @param[in] aOffset
             * Offset from which pool values are set
             */
            void
            BufferMemSet(char aValue, size_t aSize, size_t aOffset = 0);

            /**
             * @brief
             * Destroy Memory Pools If Allocated.
             */
            void
            Free();

            /**
             * @brief
             * Destroy Memory Pool If Allocated.
             */
            void
            FreeAllocations() {
                this->Free();
            }


            /**
             * @brief
             * Reset the memory pool.
             */
            void Reset() {
                mAvailable = mSize;
                mNext = 0;
                if (mpDataArray) {
                    this->BufferMemSet(0, mSize);
                }
            }
        };

        /**
         * @brief
         * A class designed mainly for device memory optimizations, the pool represents a large contiguous block of memory.
         * on an arbitrary device . Successive memory requests can be requested and passed to the requester from the pool.
         *
         */
        template<typename T>
        class MemoryHandler {
        private:
            /** Singleton instance of MemoryHandler **/
            static MemoryHandler<T>* mpInstance;
        public:
            static MemoryHandler<T>& GetInstance();

            /**
             * @brief
             * Destructor to allow correct destruction of instances created.
             */
            ~MemoryHandler() = default;

            /**
             * Singletons should not be cloneable.
             */
            MemoryHandler(MemoryHandler &aMemoryHandler) = delete;
            /**
             * Singletons should not be assignable.
             */
            void operator=(const MemoryHandler&) = delete;

            /**
             * @brief
             * The pool array getter.
             *
             * @return
             * Pointer to the pool.
             */
            MemoryUnit<T>&
            GetMemoryUnit(size_t aIdx = 0);

            /**
             * @brief
             * The pool array getter.
             *
             * @return
             * Pointer to the data array.
             */
            [[nodiscard]] const MemoryUnit<T>&
            GetMemoryUnit(size_t aIdx = 0) const {
                return mPools[aIdx];
            }

            /**
             * @brief
             * Request allocation of a specific size.
             *
             * @param[in] aSize
             * Memory Request Size
             *
             * @param[in] aIdx
             * Index of memory unit to provide allocation
             *
             * @return
             * Pointer to the data array.
             *
             */
            [[nodiscard]] T *
            RequestAllocation(size_t aSize, size_t aIdx = 0);

            /**
             * @brief
             * MemoryUnit Size getter.
             *
             * @return
             * MemoryUnit Size.
             */
            [[nodiscard]] size_t
            GetPoolSize(size_t aIdx = 0) const;

            /**
             * @brief
             * Available MemoryUnit Size getter.
             *
             * @return
             * Available MemoryUnit Size.
             */
            [[nodiscard]] size_t
            GetPoolAvailableSize(size_t aIdx = 0) const;

            /**
             * @brief
             * If MemoryUnit Allocated, return true.
             *
             * @return
             */
            [[nodiscard]] bool
            IsInitialized(size_t aIdx = 0) const;

            /**
             * @brief
             * Allocate MemoryUnit if not allocated.
             *
             * @param[in] aSize
             * Size of pool to be allocated
             *
             * @param[in] aIdx
             * index of memory unit to initialize
             */
            void
            Initialize(size_t aSize, size_t aIdx = 0);

            /**
             * @brief
             * Set Memory MemoryUnit Value.
             * @param[in] aValue
             * Value to be set
             * @param[in] aSize
             * Size of pool to be set to specified value
             * @param[in] aOffset
             * Offset from which pool values are set
             * @param[in] aIdx
             * Idx of memory unit to set
             */
            void
            BufferMemSet(char aValue, size_t aSize, size_t aOffset = 0, size_t aIdx = 0);

            /**
             * @brief
             * Destroy Memory Pools If Allocated.
             */
            void
            FreeAllocations();

            /**
             * @brief
             * Destroy Memory Pools If Allocated.
             */
            void
            FreePool(size_t aIdx = 0);

            void
            FreeMemoryUnit(size_t aIdx) {
                this->FreePool(aIdx);
            }

            /**
             * @brief
             * Returns the Memory handling strategry.
             */
            common::MemoryHandlerStrategy
            GetStrategy() {
                return common::MemoryHandlerStrategy::POOL;
            }

            /**
             * @brief
             * Reset the memory pool.
             */
            void Reset(size_t aIdx=0) {
//                mContextManager.SyncContext(aIdx);
                mPools[aIdx].Reset();
            }

            /**
             * @brief
             * destroy the singleton instance.
             */
            static void DestroyInstance() {
                if(mpInstance) {
                    mpInstance->FreeAllocations();
                    delete mpInstance;
                    mpInstance = nullptr;
                }
            }

        protected:
            /**
             * @brief
             * MemoryHandler constructor.
             *
             * @param[in] aAllocationSize
             * Initial Size of MemoryUnit
             */
            explicit MemoryHandler(size_t aAllocationSize = 0);

        private:
            /** vector of pools */
            std::vector<MemoryUnit<T>> mPools;

            /** Run context used for the data holder */
            kernels::ContextManager& mContextManager;
        };

    }
}
#endif //HCOREPP_DATA_UNITS_POOL_MEMORY_HANDLER_HPP

