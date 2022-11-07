/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCORE_HELPERS_GENERATORS_GENERATOR_HPP
#define HCORE_HELPERS_GENERATORS_GENERATOR_HPP

#include <cstdint>

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
                virtual void GenerateValues(int64_t aRowNumber, int64_t aColNumber, int64_t aLeadingDimension,
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
