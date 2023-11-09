/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_API_HCORE_HPP
#define HCOREPP_API_HCORE_HPP

#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include "hcorepp/data-units/memory-handlers/MemoryHandler.hpp"

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
            /***
             * General matrix-matrix multiplication function, C = alpha * op(A) * op(B) + beta * C.
             * @param aAlpha alpha factor
             * @param aA Input Tile A
             * @param aAOp Operation Performed on Tile A
             * @param aB Input Tile B
             * @param aBOp Operation Performed on Tile B
             * @param aBeta beta factor
             * @param aC Output Tile C
             * @param aContext hcorepp context
             * @param aFlops flops
             * @param aMemoryHandler memory handler object for allocations
             * @param aSVDArguments Compression Parameters
             * @param aCholesky flag to indicate whether gemm is being performed as part of Cholesky flow
             */
            static void
            Gemm(T aAlpha, operators::Tile<T> const &aA, blas::Op const &aAOp, operators::Tile<T> const &aB,
                 blas::Op const &aBOp, T aBeta, operators::Tile<T> &aC,
                 const hcorepp::kernels::RunContext &aContext, size_t &aFlops,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryHandler,
                 const operators::CompressionParameters &aSVDArguments = {1e-9}, bool aCholesky = false);



            /***
             * Calculate Memory Pool Size Required for Gemm to avoid successive allocations
             * @param aA Input Tile A
             * @param aB Input Tile B
             * @param aC Output Tile C
             * @param aHelpers Compression Parameters
             * @param aContext hcorepp context
             * @return
             */
            static
            size_t CalculateMemoryPoolSize(const operators::Tile<T> &aA, const operators::Tile<T> &aB,
                                           const operators::Tile<T> &aC,
                                           operators::CompressionParameters aHelpers,
                                           const kernels::RunContext &aContext);


            /***
             * Syrk operation on tiles
             * @param aAlpha alpha factor
             * @param aA Input Tile A
             * @param aAOp operation performed on A
             * @param aUplo Upper, Lower, or UpperLower
             * @param aBeta beta factor
             * @param aC C Output Tile
             * @param aContext hcorepp context
             * @param aFlops flops needed for the operation
             * @param aMemoryHandler memory unit responsible for allocations
             */
            static void
            Syrk(T aAlpha, const operators::Tile<T> &aA, const blas::Op &aAOp, blas::Uplo aUplo,
                 T aBeta, operators::Tile<T> &aC, const kernels::RunContext &aContext, size_t &aFlops,
                 dataunits::MemoryUnit<T> &aMemoryHandler);

            /***
             * Potrf operation on tiles
             * @param aA Input Tile A to be decomposed
             * @param aUplo Upper, Lower, UpperLower
             * @param aContext hcorepp context
             * @param aFlops flops needed for the operation
             * @param aMemoryHandler memory unit responsible for the allocation
             */
            static void
            Potrf(operators::Tile<T> &aA, blas::Uplo aUplo, const kernels::RunContext &aContext,
                  size_t &aFlops, dataunits::MemoryUnit<T> &aMemoryHandler);

            /***
             * TRSM operation on tiles
             * @param aSide Left, or Right
             * @param aUplo Upper, Lower, or UpperLower
             * @param aTrans Transpose operation on A
             * @param aDiag Unit or NonUnit Diagonal
             * @param aAlpha alpha factor
             * @param aA Input Tile A
             * @param aB Input Tile B
             * @param aContext hcorepp context
             * @param aFlops flops to be returned
             * @param aMemoryHandler memory unit responsible for allocations
             */
            static void
            Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                 operators::Tile<T> &aA, operators::Tile<T> &aB, const hcorepp::kernels::RunContext &aContext,
                 size_t &aFlops, hcorepp::dataunits::MemoryUnit<T> &aMemoryHandler);

        private:
            /**
             * @brief
             * Prevent Class Instantiation for API Wrapper Class.
             */
            HCore() =
            default;

            /**
             * @brief Calculate Memory MemoryUnit Size
             */
            static
            size_t CalculateGemmPoolSize(const operators::CompressedTile<T> &aC, size_t aArnk,
                                         operators::CompressionParameters aHelpers,
                                         const kernels::RunContext &aContext);
        };
    }

}

#endif //HCOREPP_API_HCORE_HPP
