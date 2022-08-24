
#ifndef HCOREPP_API_HCOREPP_HPP
#define HCOREPP_API_HCOREPP_HPP

#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>

namespace hcorepp {
    namespace api {

        /**
         * @brief
         * General matrix-matrix multiplication function, C = alpha * op(A) * op(B) + beta * C.
         *
         * @tparam T
         * Data type: float, double, std::complex<float>, or std::complex<double>
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
        template<typename T>
        void
        gemm(T aAlpha, operators::Tile<T> const &aA, blas::Op const &aAOp, operators::Tile<T> const &aB,
             blas::Op const &aBOp, T aBeta, operators::Tile<T> &aC, blas::Op const &aCOp,
             helpers::SvdHelpers &aSvdHelpers);

    }

}

#endif //HCOREPP_API_HCOREPP_HPP
