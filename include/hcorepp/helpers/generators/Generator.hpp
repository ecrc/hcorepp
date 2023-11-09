/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCORE_HELPERS_GENERATORS_GENERATOR_HPP
#define HCORE_HELPERS_GENERATORS_GENERATOR_HPP

#include <blas/util.hh>

namespace hcorepp {
    namespace helpers {
        namespace generators {
            /**
             * @brief
             * Interface for the value generators than can be used to fill values.
             *
             * @tparam T
             * The datatype for each element in the array to be filled.
             */
            template<typename T>
            class Generator {
            public:
                /**
                 * @brief
                 * Default constructor.
                 */
                Generator() = default;

                /**
                 * @brief
                 * Pure virtual function for generating the values inside a matrix,
                 * with the given specifications.
                 *
                 * @param[in] aRowNumber
                 * The number of rows.
                 *
                 * @param[in] aColNumber
                 * The number of columns.
                 *
                 * @param[in] aLeadingDimension
                 * The leading dimension for accessing the data pointer.
                 *
                 * @param[out] apData
                 * The data pointer that will be filled with values, must be correctly allocated
                 * before passing it to this function.
                 */
                virtual void GenerateValues(size_t aRowNumber, size_t aColNumber, size_t aLeadingDimension,
                                            T *apData) const = 0;

                /**
                 * @brief
                 * Default virtual destructor for inheritance.
                 */
                virtual ~Generator() = default;
            };
        }//namespace generators
    }//namespace helpers
}//namespace hcorepp

#endif //HCORE_HELPERS_GENERATORS_GENERATOR_HPP
