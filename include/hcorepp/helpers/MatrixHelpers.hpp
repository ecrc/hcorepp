/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <cstdint>
#include <vector>
#include <blas/util.hh>

#ifndef HCOREPP_HELPERS_DATA_GENERATION_HPP
#define HCOREPP_HELPERS_DATA_GENERATION_HPP

namespace hcorepp {
    namespace helpers {
        namespace matrixhelpers {

            enum TILE_COMBINATION {
                DDD,
                DDC,
                DCD,
                DCC,
                CDD,
                CDC,
                CCD,
                CCC
            };
            static const char *tile_combination_strings[] =
                    {"DDD", "DDC", "DCD", "DCC",
                     "CDD", "CDC", "CCD", "CCC"};
            enum TILE_TYPE {
                DENSE,
                COMPRESSED
            };

            template<typename T>
            void generate_dense_matrix(int64_t m, int64_t n, T *A, int64_t lda, int64_t *iseed, int64_t mode = 0,
                                       blas::real_type<T> cond = 1);

            template<typename T>
            void compress_dense_matrix(int64_t m, int64_t n, const T *A, int64_t lda, T **UV, int64_t &rk,
                                       blas::real_type<T> accuracy);

            template<typename T>
            void diff(T *Aref, int64_t lda_ref, T const *A, int64_t m, int64_t n, int64_t lda);
        }
    }
}
#endif //HCOREPP_HELPERS_DATA_GENERATION_HPP
