/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
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
            void RandomGenerator<T>::GenerateValues(size_t aRowNumber, size_t aColNumber, size_t aLeadingDimension,
                                                   T *apData) const {
                for (size_t i = 0; i < aColNumber; i++) {
                    for (size_t j = 0; j < aRowNumber; j++) {
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

