
#ifndef HCOREPP_OPERATORS_INTERFACE_TILE_HPP
#define HCOREPP_OPERATORS_INTERFACE_TILE_HPP

#include <hcorePP/data-units/DataHolder.hpp>
#include <vector>
#include "blas.hh"

using namespace hcorepp::dataunits;
using namespace std;

namespace hcorepp {
    namespace operators {

        template<typename T>
        class Tile {
        public:

            /**
             * @brief
             * Virtual destructor to allow correct destruction of concrete classes.
             */
            virtual ~Tile() = default;

            /**
             * @brief
             * Get the physical packed storage type of the tile.
             *
             * @return
             * Physical packed storage type.
             */
            blas::Uplo GetUpLoPhysical() const {
                return mUpLo;
            }

            /**
             * @brief
             * Get the logical packed storage type of the tile.
             *
             * @return
             * Logical packed storage type.
             */
            blas::Uplo GetUpLoLogical() const {
                if (mUpLo == blas::Uplo::General) {
                    return blas::Uplo::General;
                } else if ((mUpLo == blas::Uplo::Lower) == (mOperation == blas::Op::NoTrans)) {
                    return blas::Uplo::Lower;
                } else {
                    return blas::Uplo::Upper;
                }
            }

            /**
             * @brief
             * Get the physical ordering of the matrix elements in the data array.
             *
             * @return
             * Physical ordering of elements.
             */
            blas::Layout layout() const {
                return mLayout;
            }

            /**
             * @brief
             * Get sub-matrix of a tile.
             *
             * @param[in]aIndex
             * index of sub-matrix to get.
             *
             * @return
             * DataHolder object describing the Tile sub-matrix.
             */
            virtual DataHolder<T> &
            GetTileSubMatrix(size_t aIndex) = 0;

            /**
             * @brief
             * Get number of matrices describing the tile ( 1 or 2 ).
             *
             * @return
             * Number of matrices describing the tile.
             */
            virtual size_t
            GetNumberOfMatrices() = 0;

            /**
             * @brief
             * General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
             *
             * @param[in] aAlpha
             * The scalar alpha.
             * @param[in] aTileA
             * The m-by-k tile.
             * @param[in] aTileB
             * The k-by-n tile.
             * @param[in] aBeta
             * The scalar beta.
             */
            virtual void
            Gemm(T &aAlpha, DataHolder<T> const &aTileA, DataHolder<T> const &aTileB, T &aBeta) = 0;

        protected:
            blas::Op mOperation;
            blas::Uplo mUpLo;
            blas::Layout mLayout;
        };
    }
}

#endif //HCOREPP_OPERATORS_INTERFACE_TILE_HPP
