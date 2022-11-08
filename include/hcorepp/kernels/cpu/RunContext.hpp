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

            /**
             * @brief
             * Synchronizes the kernel stream.
             */
            void Sync() {
                //No need for synchronization on cpu contexts for now.
            }

            /**
             * @brief
             * Default destructor.
             */
            ~RunContext() = default;
        };
    }//namespace kernels
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CPU_RUN_CONTEXT_HPP
