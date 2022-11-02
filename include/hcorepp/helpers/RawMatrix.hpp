/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_HELPERS_RAW_MATRIX_HPP
#define HCOREPP_HELPERS_RAW_MATRIX_HPP

#include <blas/util.hh>

namespace hcorepp {
    namespace helpers {
        /**
         * @brief
         * Class encapsulating a full matrix without the tiles concept.
         *
         * @tparam T
         * The type of each element inside the matrix
         */
        template<typename T>
        class RawMatrix {
        public:
            /**
             * @brief
             * Constructor for a matrix that is initialized with random elements.
             *
             * @param[in] aM
             * The number of rows.
             *
             * @param[in] aN
             * The number of cols.
             *
             * @param[in] apSeed
             * The seed to use for random generation.
             *
             * @param[in] aMode
             * The mode to use.
             *
             * @param[in] aCond
             * The machine epsilon to use.
             */
            RawMatrix(int64_t aM, int64_t aN, int64_t *apSeed, int64_t aMode,
                      blas::real_type<T> aCond);

            /**
             * @brief
             * Constructor for an uninitialized matrix.
             *
             * @param[in] aM
             * The number of rows.
             *
             * @param[in] aN
             * The number of columns.
             */
            RawMatrix(int64_t aM, int64_t aN);

            /**
             * @brief
             * Move constructor for the raw matrix class.
             *
             * @param[in] aMatrix
             * The raw matrix that should be moved.
             */
            RawMatrix(RawMatrix &&aMatrix)  noexcept : mM(std::move(aMatrix.mM)),
                                             mN(std::move(aMatrix.mN)) {
                this->mpData = aMatrix.mpData;
                aMatrix.mpData = nullptr;
            }

            /**
             * @brief
             * Generate values inside the raw pointer randomly.
             *
             * @param[in] apSeed
             * The seed to use for random generation.
             *
             * @param[in] aMode
             * The mode to use.
             *
             * @param[in] aCond
             * The machine epsilon to use.
             */
            void GenerateValues(int64_t *apSeed, int64_t aMode,
                                blas::real_type<T> aCond);

            /**
             * @brief
             * Get the number of rows.
             *
             * @return
             * An integer representing the number of rows.
             */
            int64_t GetM() const {
                return this->mM;
            }

            /**
             * @brief
             * Get the number of columns.
             *
             * @return
             * An integer representing the number of columns.
             */
            int64_t GetN() const {
                return this->mN;
            }

            /**
             * @brief
             * Get the raw data pointer that is held inside the object.
             *
             * @return
             * A pointer to the matrix data.
             */
            const T *GetData() const {
                return this->mpData;
            }

            /**
             * @brief
             * Get the raw data pointer that is held inside the object.
             *
             * @return
             * A pointer to the matrix data.
             */
            T *GetData() {
                return this->mpData;
            }

            /**
             * @brief
             * Retrieves and computes the infinity norm of the matrix data.
             *
             * @return
             * A real value representing the norm.
             */
            blas::real_type<T> Norm();

            /**
             * @brief
             * Calculates the difference between this matrix and a given reference matrix overwriting
             * the values of this matrix.
             * m[i] = ref[i] - m[i];
             *
             * @param[in] aReferenceMatrix
             * The reference matrix to use.
             */
            void ReferenceDifference(const RawMatrix<T> &aReferenceMatrix);

            /**
             * @brief
             * Returns the memory currently occupied by the matrix data.
             *
             * @return
             * The number of bytes currently occupied by the matrix data.
             */
            size_t GetMemoryFootprint();

            /**
             * @brief
             * Creates a deep copy of this matrix.
             *
             * @return
             * A raw matrix containing its own copy of the same data as this matrix.
             */
            RawMatrix Clone();

            /**
             * @brief
             * Default destructor.
             */
            ~RawMatrix();

        private:
            /// Data pointer for the actual matrix data.
            T *mpData;
            /// The number of rows.
            int64_t mM;
            /// The number of cols.
            int64_t mN;
        };
    }//namespace helpers
}//namespace hcorepp

#endif //HCOREPP_HELPERS_RAW_MATRIX_HPP
