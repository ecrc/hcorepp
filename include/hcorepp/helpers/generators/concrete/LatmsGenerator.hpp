/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCORE_HELPERS_GENERATORS_LATMS_GENERATOR_HPP
#define HCORE_HELPERS_GENERATORS_LATMS_GENERATOR_HPP

#include <blas/util.hh>
#include <hcorepp/helpers/generators/Generator.hpp>

namespace hcorepp {
    namespace helpers {
        namespace generators {
            /**
             * @brief
             * Generator for number inside a matrix based on the LATMS routine.
             *
             * @tparam T
             * Each item inside the matrix datatype.
             */
            template<typename T>
            class LatmsGenerator : public Generator<T> {
            public:
                /**
                 * @brief
                 * Generator based on LATMS Call to generate values from eigenvalues.
                 *
                 * @param[in] apSeed
                 * The seed to use for random generation.
                 *
                 * @param[in] aMode
                 * The mode to use.
                 *
                 * @param[in] aCond
                 * The conditional value used for LATMS generation.
                 */
                LatmsGenerator(const int64_t apSeed[4], int64_t aMode,
                               blas::real_type<T> aCond);

                void GenerateValues(size_t aRowNumber, size_t aColNumber, size_t aLeadingDimension,
                                    T *apData) const override;

                /**
                 * @brief
                 * Default destructor.
                 */
                ~LatmsGenerator();

            private:
                /// Seed used for random generation
                int64_t mSeed[4];
                /// The mode used for the LATMS generation.
                int64_t mMode;
                /// The conditional value used.
                blas::real_type<T> mCond;
            };

        }//namespace generators
    }//namespace helpers
}//namespace hcorepp

#endif //HCORE_HELPERS_GENERATORS_LATMS_GENERATOR_HPP
