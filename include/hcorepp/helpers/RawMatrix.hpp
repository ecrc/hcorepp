/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_HELPERS_RAW_MATRIX_HPP
#define HCOREPP_HELPERS_RAW_MATRIX_HPP

#include <blas/util.hh>
#include "hcorepp/helpers/generators/Generator.hpp"

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
             * @param[in] aGenerator
             * Generator used to fill elements of the matrix.
             *
             */
            RawMatrix(size_t aM, size_t aN, const generators::Generator<T> &aGenerator);

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
            RawMatrix(size_t aM, size_t aN);

            /**
             * @brief
             * Constructor for an uninitialized matrix.
             *
             * @param[in] aM
             * The number of rows.
             *
             * @param[in] aN
             * The number of columns.
             *
             * @param[in] apData
             * Buffer to set as data
             */
            RawMatrix(int64_t aM, int64_t aN, T *apData);

            /**
             * @brief
             * Move constructor for the raw matrix class.
             *
             * @param[in] aMatrix
             * The raw matrix that should be moved.
             */
            RawMatrix(RawMatrix &&aMatrix) noexcept: mM(std::move(aMatrix.mM)),
                                                     mN(std::move(aMatrix.mN)) {
                this->mpData = aMatrix.mpData;
                aMatrix.mpData = nullptr;
            }

            /**
             * @brief
             * Get the number of rows.
             *
             * @return
             * An integer representing the number of rows.
             */
            size_t GetM() const {
                return this->mM;
            }

            /**
             * @brief
             * Get the number of columns.
             *
             * @return
             * An integer representing the number of columns.
             */
            size_t GetN() const {
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

            void
            Print(std::ostream &aOutputStream);

            T Normmest(T *work);

        private:
            /// Data pointer for the actual matrix data.
            T *mpData;
            /// The number of rows.
            size_t mM;
            /// The number of cols.
            size_t mN;
        };
    }//namespace helpers
}//namespace hcorepp

#endif //HCOREPP_HELPERS_RAW_MATRIX_HPP
