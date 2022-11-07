/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <hcorepp/helpers/generators/concrete/RandomGenerator.hpp>
#include <hcorepp/common/Definitions.hpp>
#include <random>

using namespace hcorepp::helpers::generators;
using namespace hcorepp::helpers;

namespace hcorepp {
    namespace helpers {
        namespace generators {

            template<typename T>
            RandomGenerator<T>::RandomGenerator() {
                srand( (unsigned)time( nullptr ) );
            }

            template<typename T>
            void RandomGenerator<T>::GenerateValues(int64_t aRowNumber, int64_t aColNumber, int64_t aLeadingDimension,
                                                   T *apData) const {
                for (int i = 0; i < aColNumber; i++) {
                    for (int j = 0; j < aRowNumber; j++) {
                        apData[i * aLeadingDimension + j] = (float) rand() / RAND_MAX;
                    }
                }
            }

            template<typename T>
            RandomGenerator<T>::~RandomGenerator() = default;

            HCOREPP_INSTANTIATE_CLASS(RandomGenerator)

        }//namespace generators
    }//namespace helpers
}//namespace hcorepp

