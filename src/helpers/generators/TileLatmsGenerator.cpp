/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/helpers/generators/concrete/TileLatmsGenerator.hpp>
#include <hcorepp/helpers/LapackWrappers.hpp>
#include <hcorepp/common/Definitions.hpp>

using namespace hcorepp::helpers::generators;
using namespace hcorepp::helpers;

namespace hcorepp {
    namespace helpers {
        namespace generators {

            template<typename T>
            TileLatmsGenerator<T>::TileLatmsGenerator(const int64_t *apSeed, int64_t aMode,
                                                      blas::real_type<T> aCond, size_t aTileSize) : mSeed{0, 0, 0, 0},
                                                                                                     mMode(aMode),
                                                                                                     mCond(aCond),
                                                                                                     mTileSize(
                                                                                                             aTileSize) {
                mSeed[0] = apSeed[0];
                mSeed[1] = apSeed[1];
                mSeed[2] = apSeed[2];
                mSeed[3] = apSeed[3];
            }

            template<typename T>
            void
            TileLatmsGenerator<T>::GenerateValues(size_t aRowNumber, size_t aColNumber, size_t aLeadingDimension,
                                                  T *apData) const {
                auto tile_size = mTileSize;
                if (tile_size <= 0) {
                    tile_size = std::min(aRowNumber, aColNumber);
                }
                // Get number of tiles in first-direction.
                auto mt = (aRowNumber / tile_size);
                if (aRowNumber % tile_size > 0) {
                    mt += 1;
                }
                // Get number of tiles in second-direction.
                auto nt = (aColNumber / tile_size);
                if (aColNumber % tile_size > 0) {
                    nt += 1;
                }
#pragma omp parallel for collapse(2) default(none) shared(nt, mt, aColNumber, aRowNumber, tile_size, apData)
                for (size_t i = 0; i < nt; i++) {
                    for (size_t j = 0; j < mt; j++) {
                        auto tile_cols = std::min(tile_size, aColNumber - i * tile_size);
                        auto tile_rows = std::min(tile_size, aRowNumber - j * tile_size);
                        auto tile_data = new T[tile_rows * tile_cols];
                        size_t min_m_n = std::min(tile_rows, tile_cols);

                        auto eigen_values = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));
                        // Exponential decay --> y = a * b^i
                        // first element i=0 --> 1   1 = a * b^0  a = 1
                        // last  element i=n --> eps eps/a = b^n  b = root_n(eps)
                        auto active_n = 80.0;
                        int active_n_floor = 80.0;
                        // Compound decay parameters
                        // If i < active_n --> first decay from 1 to eps * 10
                        // If i >= active_n --> second decay from eps * 10 to eps
                        // This is due to the notice of the real data in figure 10
                        // in the following paper https://repository.kaust.edu.sa/bitstream/handle/10754/625590/tlr-chol.pdf?sequence=1
                        auto sep = std::numeric_limits<T>::epsilon() * 10;
                        auto b_1 = pow(sep, 1.0 / active_n);
                        auto a_2 = sep;
                        auto b_2 = pow(std::numeric_limits<T>::epsilon() / a_2, 1.0 / (min_m_n - 1 - active_n_floor));
                        for (size_t i = 0; i < min_m_n; ++i) {
                            if (i < active_n_floor) {
                                eigen_values[i] = pow(b_1, i);
                            } else {
                                eigen_values[i] = a_2 * pow(b_2, i - active_n_floor);
                            }
                        }
                        T dmax = -1.0;
                        lapack_latms(
                                tile_rows, tile_cols, 'U', (int64_t *) mSeed, 'N', eigen_values,
                                mMode, mCond, dmax, tile_rows - 1, tile_cols - 1, 'N',
                                tile_data, tile_rows);
                        free(eigen_values);
                        auto st_idx = i * tile_size * aRowNumber + j * tile_size;
                        auto org_data = &apData[st_idx];
                        for (size_t jj = 0; jj < tile_cols; jj++) {
                            for (size_t ii = 0; ii < tile_rows; ii++) {
                                org_data[ii + jj * aRowNumber] = tile_data[ii + jj * tile_rows];
                            }
                        }
                        delete[] tile_data;
                    }
                }
            }

            template<typename T>
            TileLatmsGenerator<T>::~TileLatmsGenerator() = default;

            HCOREPP_INSTANTIATE_CLASS(TileLatmsGenerator)

        }//namespace generators
    }//namespace helpers
}//namespace hcorepp