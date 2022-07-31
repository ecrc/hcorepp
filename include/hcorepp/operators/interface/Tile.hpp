
#ifndef HCOREPP_OPERATORS_INTERFACE_TILE_HPP
#define HCOREPP_OPERATORS_INTERFACE_TILE_HPP

#include <vector>
#include <cstdint>
#include "blas.hh"
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/operators/helpers/SvdHelpers.hpp>

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

            blas::Op operation() const {
                return mOperation;
            }

            /**
             * @brief
             * Get matrices describing the tile.
             *
             * @return
             * vector of DataHolder object describing the Tile matrices.
             */
            virtual std::reference_wrapper<dataunits::DataHolder<T>>
            GetTileSubMatrix(size_t aIndex) = 0;

            virtual const std::reference_wrapper<dataunits::DataHolder<T>>
            GetTileSubMatrix(size_t aIndex) const = 0;

            /**
             * @brief
             * Get number of matrices describing the tile ( 1 or 2 ).
             *
             * @return
             * Number of matrices describing the tile.
             */
            virtual size_t
            GetNumberOfMatrices() const = 0;

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
            Gemm(T &aAlpha, dataunits::DataHolder<T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder<T> const &aTileB,
                 blas::Op aTileBOp, T &aBeta, int64_t ldau, int64_t Ark, const helpers::SvdHelpers &aHelpers) = 0;

            virtual int64_t
            GetTileStride(size_t aIndex) const = 0;

            virtual void
            ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                         int64_t aRank) = 0;

        protected:
            blas::Op mOperation;
            blas::Uplo mUpLo;
            blas::Layout mLayout;
            int64_t mLeadingDim;
        };
//        template class Tile<int>;
//        template class Tile<long>;
//        template class Tile<float>;
//        template class Tile<double>;

    }
}

#endif //HCOREPP_OPERATORS_INTERFACE_TILE_HPP
