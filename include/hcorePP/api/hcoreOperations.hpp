
#ifndef HCOREPP_API_HCORE_OPERATIONS_HPP
#define HCOREPP_API_HCORE_OPERATIONS_HPP


#include <hcorePP/operators/interface/Tile.hpp>
#include <hcorePP/data-units/DataHolder.hpp>

using namespace hcorepp::dataunits;
using namespace hcorepp::operators;

namespace hcorepp {
    namespace api {

        /**
         * @brief
         * General matrix-matrix multiplication function, C = alpha * op(A) * op(B) + beta * C.
         *
         * @tparam T
         * Data type: float, double, std::complex<float>, or std::complex<double>
         * @param[in] alpha
         * The scalar alpha
         * @param[in] A
         * first input tile
         * @param[in] B
         * second input tile
         * @param[in] beta
         * The scalar beta
         * @param[out] C
         * Output tile.
         *
         */
        template<typename T>
        void Gemm(T alpha, Tile<T> const &A, Tile<T> const &B, T beta, Tile<T> &C);
    }
}

#endif //HCOREPP_API_HCORE_OPERATIONS_HPP
