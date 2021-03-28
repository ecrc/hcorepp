// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_HH
#define HCORE_TEST_HH

#include "testsweeper.hh"
// #include "lapack.hh" // todo
#include "blas.hh"

class Params : public testsweeper::ParamsBase
{
public:
    Params();

    testsweeper::ParamEnum<testsweeper::DataType> datatype;
    // testsweeper::ParamEnum<lapack::Norm> norm; // todo
    // testsweeper::ParamEnum<blas::Layout> layout; // todo
    testsweeper::ParamEnum<blas::Op> transA;
    testsweeper::ParamEnum<blas::Op> transB;

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

    testsweeper::ParamDouble alpha;
    testsweeper::ParamDouble beta;
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

void gemm_test_dispatch(Params& params, bool run);
void syrk_test_dispatch(Params& params, bool run);

#endif // #ifndef HCORE_TEST_HH
