/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/kernels/ContextManager.hpp>


namespace hcorepp::kernels {

        ContextManager* ContextManager::mpInstance = nullptr;

        ContextManager& ContextManager::GetInstance() {
            if(mpInstance == nullptr) {
                mpInstance = new ContextManager();
                mpInstance->mContexts = std::vector<hcorepp::kernels::RunContext>(MAX_NUM_STREAMS);
            }

            return *mpInstance;
        }

        /**
 * @brief
 * Synchronizes the kernel stream.
 */
        void ContextManager::SyncContext(size_t aIdx) const {
            if(aIdx >= mContexts.size()) {
                throw std::runtime_error("Trying to fetch invalid Context Idx");
            }

            mContexts[aIdx].Sync();
        }

        /**
         * @brief
         * Synchronizes the main kernel stream.
         */
        void ContextManager::SyncMainContext() const {
            mContexts[0].Sync();
        }

        void ContextManager::SyncAll() const {
            for(auto& context : mContexts) {
                context.Sync();
            }
        }

        size_t ContextManager::GetNumOfContexts() const {
            return mContexts.size();
        }

        RunContext& ContextManager::GetContext(size_t aIdx) {
            if(aIdx >= mContexts.size()) {
                throw std::runtime_error("Trying to fetch invalid Context Idx");
            }

            return mContexts[aIdx];
        }

        /**
         * @brief
         * destroy the singleton instance.
         */
        void ContextManager::DestroyInstance() {
            if(mpInstance) {
                mpInstance->SyncAll();
                delete mpInstance;
                mpInstance = nullptr;
            }
        }


    }//namespace hcorepp

