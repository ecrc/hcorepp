/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
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
                // Exponential decay --> y = a * b^i
                // first element i=0 --> 1   1 = a * b^0  a = 1
                // last  element i=n --> eps eps/a = b^n  b = root_n(eps)
                auto active_n = 80.0;
                int active_n_floor = 80.0;
                // Compound decay parameters
                // If i < active_n --> first decay from 1 to eps * 10
                // If i >= active_n --> second decay from eps * 10 to eps
                // This is due to the notice of the real data in figure 10
                // in the following paper https://repository.kaust.edu.sa/bitstream/handle/10754/625590/tlr-chol.pdf?sequence=1
                auto sep = std::numeric_limits<T>::epsilon() * 10;
                auto b_1 = pow(sep, 1.0 / active_n);
                auto a_2 = sep;
                auto b_2 = pow(std::numeric_limits<T>::epsilon() / a_2, 1.0 / (min_m_n - 1 - active_n_floor));
                for (int64_t i = 0; i < min_m_n; ++i) {
                    if (i < active_n_floor) {
                        eigen_values[i] = pow(b_1, i);
                    } else {
                        eigen_values[i] = a_2 * pow(b_2, i - active_n_floor);
                    }
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