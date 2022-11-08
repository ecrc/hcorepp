/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCORE_HELPERS_GENERATORS_RANDOM_GENERATOR_HPP
#define HCORE_HELPERS_GENERATORS_RANDOM_GENERATOR_HPP

#include <hcorepp/helpers/generators/Generator.hpp>

namespace hcorepp {
    namespace helpers {
        namespace generators {
            /**
             * @brief
             * Generator for number inside a matrix generating values randomly between 0 and 1.
             *
             * @tparam T
             * Each item inside the matrix datatype.
             */
            template<typename T>
            class RandomGenerator : public Generator<T> {
            public:
                /**
                 * @brief
                 * Generator filling array with random numbers from 0 to 1.
                 */
                RandomGenerator();

                void GenerateValues(int64_t aRowNumber, int64_t aColNumber, int64_t aLeadingDimension,
                                    T *apData) const override;

                /**
                 * @brief
                 * Default destructor.
                 */
                ~RandomGenerator();
            };

        }//namespace generators
    }//namespace helpers
}//namespace hcorepp

#endif //HCORE_HELPERS_GENERATORS_RANDOM_GENERATOR_HPP
