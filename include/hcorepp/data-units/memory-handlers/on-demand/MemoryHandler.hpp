/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_DATA_UNITS_ON_DEMAND_MEMORY_HANDLER_HPP
#define HCOREPP_DATA_UNITS_ON_DEMAND_MEMORY_HANDLER_HPP

#include <functional>
#include <iostream>
#include <cstddef>
#include <unordered_set>
#include <hcorepp/common/Definitions.hpp>
#include <hcorepp/kernels/ContextManager.hpp>
#include <hcorepp/kernels/memory.hpp>
#include <hcorepp/helpers/DebuggingTimer.hpp>
#include <atomic>
#include <mutex>


namespace hcorepp {
    namespace dataunits {
        /**
         * @brief
         * A class designed mainly for the naive approach of allocating buffers on demand with no pre-allocations performed
         * Successive memory requests can be allocated. Returned buffers are not necessarily contiguous or memory-aligned
         *
         */

        template<typename T>
        class MemoryUnit {
            /** Allocated Buffers */
            std::unordered_set<T *> mAllocatedBuffers{};
            /** Context to be used for Memory Management functions for a specific pool object */
            const hcorepp::kernels::RunContext &mContext;
        public:
            explicit MemoryUnit(const hcorepp::kernels::RunContext &aContext) : mContext(aContext) {}

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
                auto *buffer = hcorepp::memory::AllocateArray<T>(aSize, mContext);
                hcorepp::memory::Memset(buffer, 0, aSize, mContext);
                mAllocatedBuffers.insert(buffer);
                return buffer;
            }

            /**
             * @brief
             * If MemoryUnit Allocated, return true.
             *
             * @return
             */
            [[nodiscard]] bool
            IsInitialized() const {
                return true;
            }

            /**
             * @brief
             * Allocate MemoryUnit if not allocated.
             *
             * @param[in] aSize
             * Size of pool to be allocated
             */
            void
            Initialize(size_t aSize) {

            }

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
            BufferMemSet(char aValue, size_t aSize, size_t aOffset = 0) {

            }

            /**
             * @brief
             * Destroy Allocated Buffers If Allocated.
             */
            void
            FreeAllocations() {
                for (auto &buff: mAllocatedBuffers) {
                    hcorepp::memory::DestroyArray(buff, mContext);
                }
                mAllocatedBuffers.clear();
            }


            /**
             * @brief
             * Frees Memory Allocations.
             */
            void Reset() {
                this->FreeAllocations();
            }

            /**
             * @brief
             * Returns the Memory handling strategy.
             */
            common::MemoryHandlerStrategy
            GetStrategy() {
                return common::MemoryHandlerStrategy::ONDEMAND;
            }
        };

        template<typename T>
        class MemoryHandler {
        private:
            /** Singleton instance of MemoryHandler **/
            static MemoryHandler<T> *mpInstance;
        public:
            static MemoryHandler<T> &GetInstance();

            /**
             * @brief
             * Destructor to allow correct destruction of instances created.
             */
            ~MemoryHandler() {
                this->FreeAllocations();
            };

            /**
             * Singletons should not be cloneable.
             */
            MemoryHandler(MemoryHandler &aMemoryHandler) = delete;

            /**
             * Singletons should not be assignable.
             */
            void operator=(const MemoryHandler &) = delete;

            [[nodiscard]] bool
            IsInitialized(size_t aIdx = 0) const {
                return mUnits[aIdx].IsInitialized();
            };

            /**
             * @brief
             * Allocate pool.
             * This function is not supported in OnDemand Memory Handler.
             */
            void
            Initialize(size_t aSize = 0, size_t aIdx = 0) {
                return mUnits[aIdx].Initialize(aSize);
            }

            /**
             * @brief
             * Request allocation of a specific size.
             *
             * @param[in] aSize
             * Memory Request Size
             *
             * @param[in] aIdx
             * Index of Memory Unit to request allocation
             *
             * @return
             * Pointer to the data array.
             */
            [[nodiscard]] T *
            RequestAllocation(size_t aSize, size_t aIdx = 0);

            /**
             * @brief
             * Set Memory Buffer Value.
             * @param[in] aValue
             * Value to be set
             * @param[in] aSize
             * Size of pool to be set to specified value
             * @param[in] aOffset
             * Offset from which pool values are set
             *
             * @param[in] aIdx
             * Index of Memory Unit to set
             */
            void
            BufferMemSet(char aValue, size_t aSize, size_t aOffset = 0, size_t aIdx = 0);

            /**
             * @brief
             * Destroy All Units.
             */
            void
            FreeAllocations();

            /**
             * @brief
             * Pool Size getter.
             *
             * @return
             * Pool Size.
             */
            [[nodiscard]] size_t
            GetPoolSize() const;

            /**
             * @brief
             * Prints the pool to the console in a readable format, useful for debugging purposes.
             */
            void Print(std::ostream &aOutStream) const;

            /**
             * @brief
             * Returns the Memory handling strategy.
             */
            common::MemoryHandlerStrategy
            GetStrategy() {
                return common::MemoryHandlerStrategy::ONDEMAND;
            }

            /**
             * @brief
             * Reset the memory pool.
             */
            void Reset(size_t aIdx=0) {
            }

            /**
             * @brief
             * destroy the singleton instance.
             */
            static void DestroyInstance() {
                delete mpInstance;
                mpInstance = nullptr;
            }

            MemoryUnit<T>&
            GetMemoryUnit(size_t aIdx = 0) {
                return this->mUnits[aIdx];
            }

            void
            FreeMemoryUnit(size_t aIdx = 0) {
                this->mUnits[aIdx].FreeAllocations();
            }

            void
            ClearUnits() {
                this->FreeAllocations();
                this->mUnits.clear();
            }

        protected:
            /**
             * @brief
             * MemoryHandler constructor.
             *
             * @param[in] aAllocationSize
             * Initial Size of Pool
             * @param[in] aContext
             * The context used to manage the data holder.
             */
            explicit MemoryHandler();

        private:
            /** Run context used for the data holder */
            kernels::ContextManager &mContextManager;
            /** Allocated Buffers */
            std::vector<MemoryUnit<T>> mUnits;
            /** Mutex for Thread-Safe Operations */
            std::mutex mMutex;
        };

    }
}
#endif //HCOREPP_DATA_UNITS_ON_DEMAND_MEMORY_HANDLER_HPP

