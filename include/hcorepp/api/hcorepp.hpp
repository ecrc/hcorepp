
#ifndef HCOREPP_API_HCOREPP_HPP
#define HCOREPP_API_HCOREPP_HPP

#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>

namespace hcorepp {
    namespace api {
        enum INPUT_TILES {
            DENSE_DENSE_DENSE,
            DENSE_DENSE_COMPRESSED,
            COMPRESSED_DENSE_DENSE,
            COMPRESSED_DENSE_COMPRESSED,
            COMPRESSED_COMPRESSED_DENSE,
            COMPRESSED_COMPRESSED_COMPRESSED,
            DENSE_COMPRESSED_DENSE,
            DENSE_COMPRESSED_COMPRESSED,
        };

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
         *
         */
        template<typename T>
        void
        gemm(T aAlpha, operators::Tile<T> const &aA, operators::Tile<T> const &aB, T aBeta, operators::Tile<T> &aC);

        template<typename T>
        void set_parameters(INPUT_TILES input, operators::Tile<T> const &A, operators::Tile<T> const &B,
                            operators::Tile<T> &C, int &num_of_rows, int &num_of_cols, int64_t &leading_dim,
                            int64_t &rank, int iteration);
    }

}

#endif //HCOREPP_API_HCOREPP_HPP
