/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/api/HCore.hpp>
#include <hcorepp/kernels/RunContext.hpp>
#include <functional>

#include <hcorepp/helpers/DebuggingTimer.hpp>
#include "hcorepp/kernels/memory.hpp"
#include "hcorepp/kernels/kernels.hpp"

using namespace hcorepp::operators;
using namespace hcorepp::dataunits;
using namespace hcorepp::kernels;
using namespace hcorepp::helpers;

namespace hcorepp {
    namespace api {

        template<typename T>
        void
        HCore<T>::Gemm(T aAlpha, const Tile <T> &aA, const blas::Op &aAOp, const Tile <T> &aB, const blas::Op &aBOp,
                       T aBeta, Tile <T> &aC, const hcorepp::kernels::RunContext &aContext, size_t &aFlops,
                       hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit,
                       const CompressionParameters &aCompressionParameters, bool aCholesky) {

            DebuggingTimer *timer = DebuggingTimer::GetDebuggingTimer();

            timer->StartSnapshot("HCore::Gemm::Initializations");

            size_t flops = 0;
            bool memoryHandlerFree = false;

            int tile_a_size = aA.GetNumOfSubMatrices();
            int tile_b_size = aB.GetNumOfSubMatrices();
            int tile_c_size = aC.GetNumOfSubMatrices();

            /// calculate total number of input tiles to determine how many intermediate Gemm functions needs to be done.
            int total_input_tiles = tile_a_size + tile_b_size;

            /// check if its Dense Dense compressed case, needs special handling only at the last iteration.
            bool dense_dense_comp = (total_input_tiles == 2 && tile_c_size == 2);

            /// controls the number of temporary gemms that will be done.
            int number_of_temporary_dense_gemms = 0;
            if (total_input_tiles == 3 || dense_dense_comp) {
                number_of_temporary_dense_gemms = 1;
            } else if (total_input_tiles == 4) {
                number_of_temporary_dense_gemms = 2;
            } else {
                number_of_temporary_dense_gemms = 0;
            }

            /// dump A dataHolders/buffers in vector.
            std::vector<DataHolder<T> *> a_data_holders;
            std::vector<blas::Op> a_data_transpose;
            a_data_holders.reserve(tile_a_size);
            a_data_transpose.reserve(tile_a_size);
            if (aA.isDense()) {
                /* New Data Holders are allocated here to allow for safe deletion at the end of the function */
                a_data_holders[0] = new DataHolder<T>(aA.GetNumOfRows(), aA.GetNumOfCols(),
                                                      aA.GetDataHolder().get().GetLeadingDim(),
                                                      aA.GetDataHolder().get().GetData(), aContext, false);
                a_data_transpose[0] = aAOp;
            } else {
                auto &comp_tile = static_cast<const CompressedTile<T> &>(aA);
                auto dh_U = new DataHolder<T>(comp_tile.GetNumOfRows(), comp_tile.GetTileRank(),
                                              comp_tile.GetULeadingDim(), comp_tile.GetUMatrix(), aContext, false);
                auto dh_V = new DataHolder<T>(comp_tile.GetTileRank(), comp_tile.GetNumOfCols(),
                                              comp_tile.GetVLeadingDim(), comp_tile.GetVMatrix(), aContext, false);
                if (aAOp == blas::Op::Trans) {
                    a_data_holders[0] = dh_V;
                    a_data_holders[1] = dh_U;
                } else {
                    a_data_holders[0] = dh_U;
                    a_data_holders[1] = dh_V;
                }
                a_data_transpose[0] = aAOp;
                a_data_transpose[1] = aAOp;
            }

            /// dump B dataHolders/buffers in vector.
            std::vector<DataHolder<T> *> b_data_holders;
            std::vector<blas::Op> b_data_transpose;
            b_data_holders.reserve(tile_b_size);
            b_data_transpose.reserve(tile_b_size);
            if (aB.isDense()) {
                /* New Data Holders are allocated here to allow for safe deletion at the end of the function */
                b_data_holders[0] = new DataHolder<T>(aB.GetNumOfRows(), aB.GetNumOfCols(),
                                                      aB.GetDataHolder().get().GetLeadingDim(),
                                                      aB.GetDataHolder().get().GetData(), aContext, false);
                b_data_transpose[0] = aBOp;
            } else {
                auto &comp_tile = static_cast<const CompressedTile<T> &>(aB);
                auto dh_U = new DataHolder<T>(comp_tile.GetNumOfRows(), comp_tile.GetTileRank(),
                                              comp_tile.GetULeadingDim(), comp_tile.GetUMatrix(), aContext, false);
                auto dh_V = new DataHolder<T>(comp_tile.GetTileRank(), comp_tile.GetNumOfCols(),
                                              comp_tile.GetVLeadingDim(), comp_tile.GetVMatrix(), aContext, false);
                if (aBOp == blas::Op::Trans) {
                    b_data_holders[0] = dh_V;
                    b_data_holders[1] = dh_U;
                } else {
                    b_data_holders[0] = dh_U;
                    b_data_holders[1] = dh_V;
                }
                b_data_transpose[0] = aBOp;
                b_data_transpose[1] = aBOp;
            }

            T alpha_local = 1;
            T beta_local = 0;
            if (dense_dense_comp) {
                alpha_local = aAlpha;
            }

            int a_idx = tile_a_size - 1;
            int b_idx = 0;

            timer->Snapshot("HCore::Gemm::Initializations");

            size_t total_size;
            size_t aRank;

            if (total_input_tiles == 3) {
                if (aB.isDense()) {
                    total_size = a_data_holders[1]->GetNumOfRows() * b_data_holders[0]->GetNumOfCols();
                    aRank = a_data_holders[0]->GetNumOfCols();
                } else {
                    total_size = a_data_holders[0]->GetNumOfRows() * b_data_holders[0]->GetNumOfCols();
                    aRank = b_data_holders[0]->GetNumOfCols();
                }
            } else if (total_input_tiles == 4) {
                total_size = a_data_holders[1]->GetNumOfRows() * b_data_holders[0]->GetNumOfCols();
                total_size += (a_data_holders[1]->GetNumOfRows() * b_data_holders[1]->GetNumOfCols());
                aRank = a_data_holders[0]->GetNumOfCols();
            } else if (dense_dense_comp) {
                total_size = a_data_holders[0]->GetNumOfRows() * b_data_holders[0]->GetNumOfCols();
                aRank = b_data_holders[0]->GetNumOfCols();
            } else {
                total_size = 0;
            }


            if (aC.isCompressed() && !dense_dense_comp) {
                size_t GemmSize = CalculateGemmPoolSize(reinterpret_cast<const CompressedTile<T> &>(aC), aRank,
                                                        aCompressionParameters, aContext);
                total_size += GemmSize;
            }

            size_t pool_idx = 0;
            if (total_size > 0 && !aMemoryUnit.IsInitialized()) {
                memoryHandlerFree = true;
                timer->StartSnapshot("HCore::Gemm::AllocateMemoryPool");
                aMemoryUnit.Initialize(total_size);
                timer->Snapshot("HCore::Gemm::AllocateMemoryPool");
            }

            if (aCholesky) {
                if (tile_a_size == 2 && tile_b_size == 2 && aAOp == blas::Op::NoTrans && aBOp == blas::Op::Trans) {
                    auto b_v = b_data_holders[0];
                    auto b_u = b_data_holders[1];
                    auto a_v = a_data_holders[1];
                    a_v->Resize(a_v->GetNumOfCols(), a_v->GetNumOfRows(), a_v->GetNumOfCols());
                    b_v->Resize(b_v->GetNumOfCols(), b_v->GetNumOfRows(), b_v->GetNumOfCols());
                    a_data_holders[1] = b_u;
                    a_data_transpose[1] = blas::Op::NoTrans;
                    b_data_holders[0] = b_v;
                    b_data_transpose[0] = blas::Op::Trans;
                    b_data_holders[1] = a_v;
                    b_data_transpose[1] = blas::Op::NoTrans;
                }
            }

            /// Resolve intermediate calls to gemm functionality through creating a temporary Dense tile for each iteration..
            std::vector<DenseTile<T> *> temp_tiles;
            while (number_of_temporary_dense_gemms > 0) {
                auto a_data = a_data_holders[a_idx];
                auto a_op = a_data_transpose[a_idx];
                auto b_data = b_data_holders[b_idx];
                auto b_op = b_data_transpose[b_idx];

                timer->StartSnapshot("HCore::Gemm::Create_temp_tile");

                size_t m;
                size_t n;
                if (a_op == blas::Op::Trans) {
                    m = a_data->GetNumOfCols();
                } else {
                    m = a_data->GetNumOfRows();
                }
                if (b_op == blas::Op::Trans) {
                    n = b_data->GetNumOfRows();
                } else {
                    n = b_data->GetNumOfCols();
                }

                temp_tiles.emplace_back(
                        new DenseTile<T>(m, n,
                                         aMemoryUnit.RequestAllocation(m * n),
                                         m,
                                         blas::Layout::ColMajor, aContext, false));

                timer->Snapshot("HCore::Gemm::Create_temp_tile");

                auto tile = temp_tiles.back();

                timer->StartSnapshot("HCore::Gemm::Apply_Gemm_on_temp_tile");
                tile->Gemm(alpha_local, *a_data, a_op, *b_data, b_op, beta_local,
                           a_data->GetLeadingDim(),
                           std::min(b_data->GetNumOfRows(), b_data->GetNumOfCols()),
                           aCompressionParameters,
                           aContext, flops, aMemoryUnit);

                timer->Snapshot(
                        "HCore::Gemm::Apply_Gemm_on_temp_tile");

                timer->StartSnapshot("HCore::Gemm::Switching_data_holders");

                if (number_of_temporary_dense_gemms == 2) {
                    DataHolder<T> &a_data_holder = (tile->GetDataHolder());
                    if (a_data_holders[a_idx] != nullptr) {
                        delete a_data_holders[a_idx];
                        a_data_holders[a_idx] = nullptr;
                    }
                    a_data_holders[a_idx] = new DataHolder<T>(a_data_holder.GetNumOfRows(),
                                                              a_data_holder.GetNumOfCols(),
                                                              a_data_holder.GetLeadingDim(), a_data_holder.GetData(),
                                                              aContext, false);
                    a_data_transpose[a_idx] = blas::Op::NoTrans;
                    b_idx++;
                } else if (number_of_temporary_dense_gemms == 1) {
                    if (a_idx == 0) {
                        DataHolder<T> &a_tile = (tile->GetDataHolder());
                        if (a_data_holders[a_idx] != nullptr) {
                            delete a_data_holders[a_idx];
                            a_data_holders[a_idx] = nullptr;
                        }

                        a_data_holders[a_idx] = new DataHolder<T>(a_tile.GetNumOfRows(), a_tile.GetNumOfCols(),
                                                                  a_tile.GetLeadingDim(), a_tile.GetData(), aContext,
                                                                  false);
                        a_data_transpose[a_idx] = blas::Op::NoTrans;
                        b_idx++;
                    } else {
                        DataHolder<T> &b_tile = tile->GetDataHolder();
                        if (b_data_holders[b_idx] != nullptr) {
                            delete b_data_holders[b_idx];
                            b_data_holders[b_idx] = nullptr;
                        }

                        b_data_holders[b_idx] = new DataHolder<T>(b_tile.GetNumOfRows(), b_tile.GetNumOfCols(),
                                                                  b_tile.GetLeadingDim(), b_tile.GetData(), aContext,
                                                                  false);
                        b_data_transpose[b_idx] = blas::Op::NoTrans;
                        a_idx--;
                    }
                }
                number_of_temporary_dense_gemms--;

                timer->Snapshot("HCore::Gemm::Switching_data_holders");
                pool_idx += (a_data->GetNumOfRows() * b_data->GetNumOfCols());
            }


            /// the last iteration needs to be resolved according to the Tile type either Compressed or Dense.
            if (number_of_temporary_dense_gemms == 0) {
                ///Dense dense compressed case special


                if (dense_dense_comp) {
                    timer->StartSnapshot("HCore::Gemm::Resolving_final_DDC_Gemm");
                    auto target = temp_tiles[0];
                    auto a_op = blas::Op::NoTrans;
                    auto b_op = blas::Op::NoTrans;
                    alpha_local = 1;
                    auto &comp_tile = static_cast<const CompressedTile<T> &>(aC);
                    auto dh_U = DataHolder<T>(comp_tile.GetNumOfRows(), comp_tile.GetTileRank(),
                                              comp_tile.GetULeadingDim(), comp_tile.GetUMatrix(), aContext, false);
                    auto dh_V = DataHolder<T>(comp_tile.GetTileRank(), comp_tile.GetNumOfCols(),
                                              comp_tile.GetVLeadingDim(), comp_tile.GetVMatrix(), aContext, false);

                    // W += beta * Cu*Cv;

                    target->Gemm(aBeta, dh_U, a_op, dh_V, b_op, alpha_local,
                                 dh_U.GetLeadingDim(),
                                 std::min(dh_V.GetNumOfRows(), dh_V.GetNumOfCols()),
                                 aCompressionParameters, aContext, flops, aMemoryUnit);

                    size_t num_of_rows = target->GetDataHolder().get().GetNumOfRows();
                    size_t num_of_cols = target->GetDataHolder().get().GetNumOfCols();

                    int64_t c_rank = -1;

                    // todo :: Revisit the Handling of DDC case. This is not handled correctly
                    aC.ReadjustTile(num_of_rows, num_of_cols, target->GetTileSubMatrix(0),
                                    num_of_rows, c_rank, aContext);
                    timer->Snapshot("HCore::Gemm::Resolving_final_DDC_Gemm");
                } else {
                    timer->StartSnapshot("HCore::Gemm::Resolving_final_Gemm");
                    auto a_data = a_data_holders[a_idx];
                    auto b_data = b_data_holders[b_idx];


                    size_t c_rank;
                    c_rank = a_data->GetNumOfCols();
                    aC.Gemm(aAlpha, *a_data, a_data_transpose[a_idx], *b_data, b_data_transpose[b_idx], aBeta,
                            a_data->GetLeadingDim(),
                            c_rank, aCompressionParameters, aContext, flops, aMemoryUnit, aCholesky);

                    timer->Snapshot("HCore::Gemm::Resolving_final_Gemm");
                }
            }


            if (memoryHandlerFree) {
                timer->StartSnapshot("HCore::Gemm::DestroyMemoryPool");
                aMemoryUnit.FreeAllocations();
                timer->Snapshot("HCore::Gemm::DestroyMemoryPool");
            } else {
                aMemoryUnit.Reset();
            }


            timer->StartSnapshot("HCore::Gemm::Destroy_temp_tiles");

            /// free the allocated temporary tiles
            for (auto tile: temp_tiles) {
                delete tile;
            }

            for (auto holder: a_data_holders) {
                delete holder;
            }

            for (auto holder: b_data_holders) {
                delete holder;
            }

            timer->Snapshot("HCore::Gemm::Destroy_temp_tiles");

            aFlops += flops;
        }

        template<typename T>
        size_t HCore<T>::CalculateGemmPoolSize(const CompressedTile <T> &aC, size_t aARank,
                                               const CompressionParameters aHelpers,
                                               const kernels::RunContext &aContext) {
            size_t m = aC.GetNumOfRows();
            size_t n = aC.GetNumOfCols();
            size_t ldcu = aC.GetULeadingDim();
            size_t Crk = std::min(m, n);

            size_t Um = m;
            size_t Un = aARank + Crk;
            size_t min_Um_Un = std::min(Um, Un);

            size_t Vm = n;
            size_t Vn = aARank + Crk;

            size_t min_Vm_Vn = std::min(Vm, Vn);

            size_t max_rows;
            size_t max_cols;
            if (aHelpers.GetUngqr()) {
                max_rows = min_Um_Un;
                max_cols = min_Vm_Vn;
            } else {
                max_rows = Um;
                max_cols = Vm;
            }

            size_t sizeS;
            if (aHelpers.GetTrmm()) {
                sizeS = min_Um_Un;
            } else {
                sizeS = std::min(m, n);
                sizeS = std::min(sizeS, (aARank + Crk));
            }

            size_t host_size = 0;

            size_t U_size = Um * Un;
            size_t Utau_size = min_Um_Un;
            size_t RU_size = min_Um_Un * Un;
            size_t V_size = Vm * Vn;
            size_t Vtau_size = min_Vm_Vn;
            size_t Unew_size = max_rows * sizeS;
            size_t VTnew_size = sizeS * max_cols;
            size_t UV_size = (ldcu + n) * sizeS;
            size_t workspace_size = hcorepp::kernels::HCoreKernels<T>::CalculateGemmWorkspaceSize(Um, Un, Vm, Vn,
                                                                                                   sizeS,
                                                                                                   aHelpers, host_size,
                                                                                                   aContext);
            size_t RV_size = 0;
            size_t RURV_size = 0;
            size_t Vnew_size = 0;

            size_t total_size = U_size + Utau_size + RU_size + V_size + Vtau_size + Unew_size + VTnew_size + UV_size +
                                workspace_size;

            if (!aHelpers.GetTrmm()) {
                RV_size = min_Vm_Vn * Vn;
                RURV_size = min_Um_Un * min_Vm_Vn;
                total_size += RV_size + RURV_size;
            }

            if (aHelpers.GetUngqr()) {
                Vnew_size = Vm * sizeS;
                total_size += Vnew_size;
            }

            return total_size;
        }

        template<typename T>
        size_t HCore<T>::CalculateMemoryPoolSize(const Tile <T> &aA, const Tile <T> &aB,
                                                 const Tile <T> &aC,
                                                 const operators::CompressionParameters aHelpers,
                                                 const RunContext &aContext) {
            size_t total_size = 0;
            int tile_a_size = aA.GetNumOfSubMatrices();
            int tile_b_size = aB.GetNumOfSubMatrices();
            int tile_c_size = aC.GetNumOfSubMatrices();

            /// calculate total number of input tiles to determine how many intermediate Gemm functions needs to be done.
            int total_input_tiles = tile_a_size + tile_b_size;

            /// check if its Dense Dense compressed case, needs special handling only at the last iteration.
            bool dense_dense_comp = (total_input_tiles == 2 && tile_c_size == 2);

            /// controls the number of temporary gemms that will be done.
            int number_of_temporary_dense_gemms = 0;
            if (total_input_tiles == 3 || dense_dense_comp) {
                number_of_temporary_dense_gemms = 1;
            } else if (total_input_tiles == 4) {
                number_of_temporary_dense_gemms = 2;
            } else {
                number_of_temporary_dense_gemms = 0;
            }

            size_t aRank;

            if (total_input_tiles == 3) {
                if (aB.isDense()) {
                    auto &comp_tile = static_cast<const CompressedTile<T> &>(aA);
                    total_size =
                            comp_tile.GetTileRank() * aB.GetNumOfCols();
                    aRank = comp_tile.GetTileRank();
                } else {
                    auto &comp_tile = static_cast<const CompressedTile<T> &>(aB);
                    total_size =
                            aA.GetNumOfRows() * comp_tile.GetTileRank();
                    aRank = comp_tile.GetTileRank();
                }
            } else if (total_input_tiles == 4) {
                auto &comp_tileA = static_cast<const CompressedTile<T> &>(aA);
                auto &comp_tileB = static_cast<const CompressedTile<T> &>(aB);
                total_size = comp_tileA.GetTileRank() * comp_tileB.GetTileRank();
                total_size += (comp_tileA.GetTileRank() *
                               comp_tileB.GetTileRank());
                aRank = comp_tileA.GetTileRank();
            } else if (dense_dense_comp) {
                total_size = aA.GetNumOfRows() * aB.GetNumOfCols();
                aRank = aB.GetNumOfCols();
            } else {
                total_size = 0;
            }

            if (aC.isCompressed() && !dense_dense_comp) {
                size_t GemmSize = CalculateGemmPoolSize(reinterpret_cast<const CompressedTile<T> &>(aC), aRank,
                                                        aHelpers, aContext);
                total_size += GemmSize;
            }

            return total_size;


        }

        template<typename T>
        void
        HCore<T>::Syrk(T aAlpha, const operators::Tile<T> &aA, const blas::Op &aAOp, const blas::Uplo aUplo,
                       T aBeta, operators::Tile<T> &aC, const kernels::RunContext &aContext, size_t &aFlops,
                       dataunits::MemoryUnit<T> &aMemoryUnit) {

            if (aA.GetNumOfSubMatrices() == 2) {
                /// compression parameters and rank are not used since all tiles used are dense.
                CompressionParameters aCompressionParameters;
                auto rank = 1;

                auto &comp_tile = static_cast<const CompressedTile<T> &>(aA);

                bool memoryHandlerFree = false;
                size_t total_size = 0;
                {
                    total_size += comp_tile.GetTileRank() * comp_tile.GetTileRank();
                    total_size += aA.GetNumOfRows() * aA.GetTileRank();
                    if (aC.isCompressed()) {
                        auto &c_compressed = static_cast<const CompressedTile<T> &>(aC);
                        size_t GemmSize = CalculateGemmPoolSize(reinterpret_cast<const CompressedTile<T> &>(aC),
                                                                c_compressed.GetTileRank(),
                                                                aCompressionParameters, aContext);
                        total_size += GemmSize;
                    }
                }
                if (total_size > 0 && !aMemoryUnit.IsInitialized()) {
                    memoryHandlerFree = true;
                    aMemoryUnit.Initialize(total_size);
                }

                auto dh_U = new DataHolder<T>(comp_tile.GetTileRank(), comp_tile.GetNumOfRows(),
                                              comp_tile.GetULeadingDim(), comp_tile.GetUMatrix(), aContext, false);
                auto dh_V = new DataHolder<T>(comp_tile.GetNumOfCols(), comp_tile.GetTileRank(),
                                              comp_tile.GetVLeadingDim(), comp_tile.GetVMatrix(), aContext, false);
                T alpha = 1;
                T beta = 0;

                auto rows = dh_U->GetNumOfRows();
                auto cols = dh_V->GetNumOfCols();
                auto leading_dim = dh_V->GetLeadingDim();
                auto layout = aA.GetLayout();

                DenseTile<T> *Av_AvT_tile = new DenseTile<T>(rows, cols, aMemoryUnit.RequestAllocation(rows * cols),
                                                             leading_dim, layout, aContext,false);

                auto a_data = dh_V;
                auto b_data = dh_V;

                Av_AvT_tile->Gemm(alpha, *a_data, blas::Op::Trans, *b_data, blas::Op::NoTrans, beta, leading_dim,
                                  rank, aCompressionParameters, aContext, aFlops, aMemoryUnit);

                rows = aA.GetNumOfRows();
                cols = aA.GetTileRank();
                leading_dim = rows;

                DenseTile<T> *Au_Av_AvT_tile = new DenseTile<T>(rows, cols, aMemoryUnit.RequestAllocation(rows * cols),
                                                                leading_dim, layout, aContext,false);
                auto au_data = dh_U;
                auto av_avt_data = Av_AvT_tile->GetDataHolder();

                rank = aA.GetTileRank();
                Au_Av_AvT_tile->Gemm(alpha, *au_data, blas::Op::NoTrans, av_avt_data, blas::Op::NoTrans, beta,
                                     leading_dim, rank, aCompressionParameters, aContext, aFlops, aMemoryUnit);

                auto au_av_avt_data = Au_Av_AvT_tile->GetDataHolder();
                auto aut_data = dh_U;

                T alpha1 = -1.0;

                aC.Gemm(alpha1, au_av_avt_data, blas::Op::NoTrans, *aut_data, blas::Op::Trans, aBeta,
                        leading_dim, rank, aCompressionParameters, aContext, aFlops, aMemoryUnit);

                if (aUplo == blas::Uplo::Lower) {
                    HCoreKernels<T>::FillMatrixTriangle(blas::Uplo::Upper, aC.GetNumOfRows(), aC.GetNumOfCols(),
                                                        aC.GetDataHolder().get().GetData(),
                                                        aC.GetLayout(), beta, aContext);
                } else if (aUplo == blas::Uplo::Upper) {
                    HCoreKernels<T>::FillMatrixTriangle(blas::Uplo::Lower, aC.GetNumOfRows(), aC.GetNumOfCols(),
                                                        aC.GetDataHolder().get().GetData(),
                                                        aC.GetLayout(), beta, aContext);
                }

                if (memoryHandlerFree) {
                    aMemoryUnit.FreeAllocations();
                } else {
                    aMemoryUnit.Reset();
                }

                delete dh_U;
                delete dh_V;
                delete Av_AvT_tile;
                delete Au_Av_AvT_tile;
            } else if (aA.GetNumOfSubMatrices() == 1) {
                HCoreKernels<T>::syrk(aC.GetLayout(), aUplo, aAOp, aC.GetNumOfRows(), aC.GetNumOfCols(),
                                      aAlpha, aA.GetDataHolder().get().GetData(),
                                      aA.GetDataHolder().get().GetLeadingDim(),
                                      aBeta, aC.GetDataHolder().get().GetData(),
                                      aC.GetDataHolder().get().GetLeadingDim(),
                                      aContext);
            }
        }

        template<typename T>
        void HCore<T>::Potrf(operators::Tile<T> &aA, const blas::Uplo aUplo, const kernels::RunContext &aContext,
                             size_t &aFlops, dataunits::MemoryUnit<T> &aMemoryUnit) {

            if (aA.GetNumOfSubMatrices() != 1) {
                throw std::runtime_error(" Potrf works only with dense tiles");
            }

            size_t host_size = 0;
            bool memoryHandlerFree = false;
            size_t workspace_size = HCoreKernels<T>::CalculatePotrfWorkspaceSize(aA.GetDataHolder().get().GetData(),
                                                                                 aUplo,
                                                                                 aA.GetNumOfRows(),
                                                                                 aA.GetDataHolder().get().GetLeadingDim(),
                                                                                 host_size, aContext);

            if (workspace_size > 0 && !aMemoryUnit.IsInitialized()) {
                memoryHandlerFree = true;
                aMemoryUnit.Initialize(workspace_size);
            }

            T *workspace = nullptr;
            if (workspace_size > 0) {
                workspace = aMemoryUnit.RequestAllocation(workspace_size);
            }

            HCoreKernels<T>::potrf(aUplo, workspace, workspace_size, workspace_size, aA.GetNumOfRows(),
                                   aA.GetDataHolder().get().GetData(), aA.GetDataHolder().get().GetLeadingDim(),
                                   aA.GetLayout(), aContext);

            if (memoryHandlerFree) {
                aMemoryUnit.FreeAllocations();
            } else {
                aMemoryUnit.Reset();
            }

        }

        template<typename T>
        void HCore<T>::Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                            operators::Tile<T> &aA, operators::Tile<T> &aB, const kernels::RunContext &aContext,
                            size_t &aFlops, dataunits::MemoryUnit<T> &aMemoryHandler) {

            if (aB.GetNumOfSubMatrices() != 2) {
                throw std::runtime_error(" TRSM: Tile B must be compressed ");
            }

            auto &comp_tile = static_cast<const CompressedTile<T> &>(aB);
            auto dh_U = new DataHolder<T>(comp_tile.GetNumOfRows(), comp_tile.GetTileRank(),
                                          comp_tile.GetULeadingDim(), comp_tile.GetUMatrix(), aContext, false);
            auto dh_V = new DataHolder<T>(comp_tile.GetTileRank(), comp_tile.GetNumOfCols(),
                                          comp_tile.GetVLeadingDim(), comp_tile.GetVMatrix(), aContext, false);

            HCoreKernels<T>::trsm(aB.GetLayout(), aSide, aUplo, aTrans, aDiag, aB.GetNumOfRows(),
                                  comp_tile.GetTileRank(),
                                  aAlpha, aA.GetDataHolder().get().GetData(),
                                  aA.GetDataHolder().get().GetLeadingDim(), dh_V->GetData(),
                                  aB.GetLeadingDim(), aContext);

            delete dh_U;
            delete dh_V;

        }

        HCOREPP_INSTANTIATE_CLASS(HCore)

    }
}
