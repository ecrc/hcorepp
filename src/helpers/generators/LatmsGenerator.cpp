/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <hcorepp/helpers/generators/concrete/LatmsGenerator.hpp>
#include <hcorepp/helpers/LapackWrappers.hpp>
#include <hcorepp/common/Definitions.hpp>

using namespace hcorepp::helpers::generators;
using namespace hcorepp::helpers;

namespace hcorepp {
    namespace helpers {
        namespace generators {

            template<typename T>
            LatmsGenerator<T>::LatmsGenerator(const int64_t *apSeed, int64_t aMode,
                                              blas::real_type<T> aCond) : mSeed{0, 0, 0, 0},
                                                                          mMode(aMode), mCond(aCond) {
                mSeed[0] = apSeed[0];
                mSeed[1] = apSeed[1];
                mSeed[2] = apSeed[2];
                mSeed[3] = apSeed[3];
            }

            template<typename T>
            void LatmsGenerator<T>::GenerateValues(int64_t aRowNumber, int64_t aColNumber, int64_t aLeadingDimension,
                                                   T *apData) const {
                int64_t min_m_n = std::min(aRowNumber, aColNumber);

                auto eigen_values = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));
                double deno = (min_m_n - 1);
                for (int64_t i = 0; i < min_m_n; ++i) {
                    eigen_values[i] = (deno - i  + i * 1e-16) / deno;
                }
                T dmax = -1.0;
                lapack_latms(
                        aRowNumber, aColNumber, 'U', (int64_t *) mSeed, 'N', eigen_values, mMode, mCond,
                        dmax, aRowNumber - 1, aColNumber - 1, 'N',
                        apData, aLeadingDimension);
                free(eigen_values);
            }

            template<typename T>
            LatmsGenerator<T>::~LatmsGenerator() = default;

            HCOREPP_INSTANTIATE_CLASS(LatmsGenerator)

        }//namespace generators
    }//namespace helpers
}//namespace hcorepp