/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_CONTEXT_MANAGER_HPP
#define HCOREPP_KERNELS_CONTEXT_MANAGER_HPP

#include <hcorepp/kernels/RunContext.hpp>
#define MAX_NUM_STREAMS 50

namespace hcorepp {
    namespace kernels {
        /**
         * @brief
         * Singleton Class responsible for managing and delivering all HCorePP Context Objects.
         */
        class ContextManager {
        private:
            /** Singleton instance of MemoryHandler **/
            static ContextManager* mpInstance;
        public:
            static ContextManager& GetInstance();

            /**
             * @brief
             * Destructor to allow correct destruction of instances created.
             */
            ~ContextManager() = default;

            /**
             * Singletons should not be cloneable.
             */
            ContextManager(ContextManager &) = delete;
            /**
             * Singletons should not be assignable.
             */
            void operator=(const ContextManager&) = delete;

            /**
             * @brief
             * Synchronizes the kernel stream.
             */
            void SyncContext(size_t aIdx = 0) const;

            /**
             * @brief
             * Synchronizes the main kernel stream.
             */
            void SyncMainContext() const;

            void SyncAll() const;

            [[nodiscard]] size_t GetNumOfContexts() const;

            RunContext& GetContext(size_t aIdx = 0);
            /**
             * @brief
             * destroy the singleton instance.
             */
            static void DestroyInstance();

        protected:
            /**
             * @brief
             * Context Manager constructor.
             *
             */
            ContextManager() = default;

        private:
            std::vector<hcorepp::kernels::RunContext> mContexts;

        };
    }//namespace kernels
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CONTEXT_MANAGER_HPP
