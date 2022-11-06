/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
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
                                                      blas::real_type<T> aCond, int64_t aTileSize) : mSeed{0, 0, 0, 0},
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
            TileLatmsGenerator<T>::GenerateValues(int64_t aRowNumber, int64_t aColNumber, int64_t aLeadingDimension,
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
                for (int i = 0; i < nt; i++) {
                    for (int j = 0; j < mt; j++) {
                        auto tile_cols = std::min(tile_size, aColNumber - i * tile_size);
                        auto tile_rows = std::min(tile_size, aRowNumber - j * tile_size);
                        auto tile_data = new T[tile_rows * tile_cols];
                        int64_t min_m_n = std::min(tile_rows, tile_cols);

                        auto eigen_values = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));

                        double deno = (min_m_n - 1);
                        for (int64_t i = 0; i < min_m_n; ++i) {
                            eigen_values[i] = (deno - i  + i * 1e-16) / deno;
                        }
                        T dmax = -1.0;
                        lapack_latms(
                                tile_rows, tile_cols, 'U', (int64_t *) mSeed, 'N', eigen_values,
                                mMode, mCond, dmax, tile_rows - 1, tile_cols - 1, 'N',
                                tile_data, tile_rows);
                        free(eigen_values);
                        auto st_idx = i * tile_size * aRowNumber + j * tile_size;
                        auto org_data = &apData[st_idx];
                        for (int jj = 0; jj < tile_cols; jj++) {
                            for (int ii = 0; ii < tile_rows; ii++) {
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