// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_TEST_HH
#define HCORE_TEST_TEST_HH

#include "testsweeper.hh"
#include "blas.hh"

class Params : public testsweeper::ParamsBase
{
public:
    Params();

    testsweeper::ParamEnum<testsweeper::DataType> datatype;

    testsweeper::ParamEnum<blas::Layout> layout;
    testsweeper::ParamEnum<blas::Uplo>   uplo;
    testsweeper::ParamEnum<blas::Side>   side;
    testsweeper::ParamEnum<blas::Diag>   diag;
    testsweeper::ParamEnum<blas::Op>     trans;
    testsweeper::ParamEnum<blas::Op>     transA;
    testsweeper::ParamEnum<blas::Op>     transB;
    testsweeper::ParamEnum<blas::Op>     transC;

    testsweeper::ParamChar check;
    testsweeper::ParamChar use_trmm;
    testsweeper::ParamChar use_ungqr;
    testsweeper::ParamChar truncate_with_tol;

    testsweeper::ParamInt3 dim;

    testsweeper::ParamInt align;
    testsweeper::ParamInt repeat;
    testsweeper::ParamInt verbose;
    testsweeper::ParamInt latms_mode;
    testsweeper::ParamInt truncate_with_fixed_rk;

    testsweeper::ParamComplex alpha;
    testsweeper::ParamComplex beta;

    testsweeper::ParamDouble tol;
    testsweeper::ParamDouble time;
    testsweeper::ParamDouble gflops;
    testsweeper::ParamDouble gbytes;
    testsweeper::ParamDouble ref_time;
    testsweeper::ParamDouble ref_gflops;
    testsweeper::ParamDouble ref_gbytes;

    testsweeper::ParamScientific latms_cond;
    testsweeper::ParamScientific accuracy;
    testsweeper::ParamScientific error;

    testsweeper::ParamString rk;

    testsweeper::ParamOkay okay;

    std::string routine;

}; // class Params

namespace hcore {
namespace test {

// BLAS 3
void gemm_dispatch(Params& params, bool run);
void syrk_dispatch(Params& params, bool run);
void trsm_dispatch(Params& params, bool run);

// LAPACK
void potrf_dispatch(Params& params, bool run);

} // namespace test
} // namespace hcore

#endif // #ifndef HCORE_TEST_TEST_HH
