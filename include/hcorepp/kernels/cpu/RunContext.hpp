/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_CPU_RUN_CONTEXT_HPP
#define HCOREPP_KERNELS_CPU_RUN_CONTEXT_HPP

namespace hcorepp {
    namespace kernels {
        /**
         * @brief
         * Class responsible of providing the run context for the CPU device.
         */
        class RunContext {
        public:
            /**
             * @brief
             * Default Constructor.
             */
            RunContext() = default;

            RunContext ForkChildContext() {
                RunContext context;
                return context;
            }

            /**
             * @brief
             * Synchronizes the kernel stream.
             */
            void Sync() const {
                //No need for synchronization on cpu contexts for now.
            }

            /**
             * @brief
             * Default destructor.
             */
            ~RunContext() = default;

            /**
             * @brief
             * Check if Context supports OMP Parallelization
             */
             [[nodiscard]] bool SupportsOMP() const {
                 return true;
             }


        };
    }//namespace kernels
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CPU_RUN_CONTEXT_HPP
