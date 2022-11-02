/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_API_HCOREPP_HPP
#define HCOREPP_API_HCOREPP_HPP

#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>

namespace hcorepp {
    namespace api {
        /**
         * @brief
         * High-Level Wrapper class containing the static API for Hcore++ operations.
         *
         * @tparam T
         * Data type: float, double, or complex
         */
        template<typename T>
        class HCore {
        public:
            /**
             * @brief
             * General matrix-matrix multiplication function, C = alpha * op(A) * op(B) + beta * C.
             *
             * @param[in] aAlpha
             * The scalar alpha
             * @param[in] aA
             * first input tile
             * @param[in] aB
             * second input tile
             * @param[in] aBeta
             * The scalar beta
             * @param[out] aC
             * Output tile.
             * @param[in] aSvdHelpers
             * SVD Helpers class reference.
             *
             */
            static void
            Gemm(T aAlpha, operators::Tile<T> const &aA, blas::Op const &aAOp, operators::Tile<T> const &aB,
                 blas::Op const &aBOp, T aBeta, operators::Tile<T> &aC,
                 const operators::SVDParameters &aSVDArguments = {1e-9});

        private:
            /**
             * @brief
             * Prevent Class Instantiation for API Wrapper Class.
             */
            HCore() = default;
        };

    }

}

#endif //HCOREPP_API_HCOREPP_HPP
