// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "test.hh"
#include "hcore/hcore.hh"

#include "blas.hh"
#include "lapack.hh"
#include "testsweeper.hh"

#include <ctime>
#include <cstdio>
#include <string>
#include <cassert>

enum Section {
    newline=0,
    chol,
    blas3_gemm,
    blas3_syrk,
    blas3_trsm,
    num_sections,
};

const char* section_names[] = {
    "",
    "Cholesky",
    "Level 3 BLAS -- GEMM",
    "Level 3 BLAS -- SYRK",
    "Level 3 BLAS -- TRSM",
    
};

std::vector<testsweeper::routines_t> routines = {
    { "potrf",    hcore::test::potrf_dispatch, Section::chol       },
    // { "potrf_c",  hcore::test::potrf_dispatch, Section::chol       }, // todo
    // { "potrf_d",  hcore::test::potrf_dispatch, Section::chol       }, // todo
    { "",         nullptr,                     Section::newline    },

    { "gemm_ddd", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "gemm_ddc", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "gemm_dcd", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "gemm_dcc", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "gemm_cdd", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "gemm_cdc", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "gemm_ccd", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "gemm_ccc", hcore::test::gemm_dispatch,  Section::blas3_gemm },
    { "",         nullptr,                     Section::newline    },

    { "syrk_dd",  hcore::test::syrk_dispatch,  Section::blas3_syrk },
    { "syrk_dc",  hcore::test::syrk_dispatch,  Section::blas3_syrk },
    { "syrk_cd",  hcore::test::syrk_dispatch,  Section::blas3_syrk },
    { "syrk_cc",  hcore::test::syrk_dispatch,  Section::blas3_syrk },
    { "",         nullptr,                     Section::newline    },

    { "trsm",  hcore::test::trsm_dispatch,  Section::blas3_trsm },
    // { "trsm_dd",  hcore::test::trsm_dispatch,  Section::blas3_trsm }, // todo
    // { "trsm_dc",  hcore::test::trsm_dispatch,  Section::blas3_trsm }, // todo
    // { "trsm_cd",  hcore::test::trsm_dispatch,  Section::blas3_trsm }, // todo
    // { "trsm_cc",  hcore::test::trsm_dispatch,  Section::blas3_trsm }, // todo
    { "",         nullptr,                     Section::newline    },
};

Params::Params():
    ParamsBase(),
    datatype(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "type", 4, testsweeper::ParamType::List, testsweeper::DataType::Double,
        testsweeper::char2datatype, testsweeper::datatype2char,
        testsweeper::datatype2str,
        "one of: s, r32, single, float; d, r64, double (default); "
                "c, c32, 'complex<float>'; z, c64, 'complex<double>'"),
    layout(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "layout", 6, testsweeper::ParamType::List, blas::Layout::ColMajor,
        blas::char2layout, blas::layout2char, blas::layout2str,
        "layout: r=row major, c=column major (default)"),
    uplo(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "uplo", 6, testsweeper::ParamType::List, blas::Uplo::Lower,
        blas::char2uplo, blas::uplo2char, blas::uplo2str,
        "triangle: l=lower, u=upper"),
    side(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "side", 6, testsweeper::ParamType::List, blas::Side::Left,
        blas::char2side, blas::side2char, blas::side2str,
        "side: l=left, r=right"),
    diag(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "diag", 7, testsweeper::ParamType::List, blas::Diag::NonUnit,
        blas::char2diag, blas::diag2char, blas::diag2str,
        "diagonal: n=non-unit, u=unit"),
    trans(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "trans", 7, testsweeper::ParamType::List, blas::Op::NoTrans,
        blas::char2op, blas::op2char, blas::op2str,
        "transpose: n=notrans (default), t=trans, c=conjtrans"),
    transA(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "transA", 7, testsweeper::ParamType::List, blas::Op::NoTrans,
        blas::char2op, blas::op2char, blas::op2str,
        "transpose of A: n=notrans (default), t=trans, c=conjtrans"),
    transB(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "transB", 7, testsweeper::ParamType::List, blas::Op::NoTrans,
        blas::char2op, blas::op2char, blas::op2str,
        "transpose of B: n=notrans (default), t=trans, c=conjtrans"),
    transC(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "transC", 7, testsweeper::ParamType::List, blas::Op::NoTrans,
        blas::char2op, blas::op2char, blas::op2str,
        "transpose of C: n=notrans (default), t=trans, c=conjtrans"),
    dim(
        // name, width, type, min, max, help
        "dim", 5, testsweeper::ParamType::List, 0, 1000000,
        "m by n by k dimensions"),
    align(
        // name, width, type, min, max, default, help
        "align", 0, testsweeper::ParamType::List, 1, 1, 1024,
        "column alignment (sets lda, ldb, etc. to multiple of align)" ),
    latms_mode(
        // name, width, type, default, min, max, help
        "latms_mode", 0, testsweeper::ParamType::Value, 0, 0, 6,
        "pass to lapacke_latms to describe how the singular/eigenvalues "
        "are to be specified (0 to 6)"),
    alpha(
        // name, width, precision, type, default, min, max, help
        "alpha", 6, 4, testsweeper::ParamType::List,
        "3.141592653589793+1.414213562373095i",
        -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(), "alpha value"),
    beta(
        "beta", 6, 4, testsweeper::ParamType::List,
        "2.718281828459045+1.732050807568877i",
        -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(), "beta value"),
    check(
        // name, width, type, default, valid, help
        "check", 0, testsweeper::ParamType::Value, 'y', "ny",
        "check the results (default yes)"),
    use_gemm(
        // name, width, type, default, valid, help
        "use_gemm", 0, testsweeper::ParamType::Value, 'y', "ny",
        "use gemm with ungqr (default yes)"),
    truncate_with_tol(
        // name, width, type, default, valid, help
        "truncate_with_tol", 0, testsweeper::ParamType::Value, 'n', "ny",
        "truncation with tolerance * accuracy (default no"),
    norm(
        // name, width, type, default, char2enum, enum2char, enum2str, help
        "norm", 0, testsweeper::ParamType::Value, lapack::Norm::Inf,
        lapack::char2norm, lapack::norm2char, lapack::norm2str,
        "norm: o=one, 2=two, i=inf, f=fro, m=max"),
    truncate_with_fixed_rk(
        // name, width, type, default, min, max, help
        "truncate_with_fixed_rk", 0, testsweeper::ParamType::Value,
        0, 0, 1000000, "truncation with fixed rank (default 0 (no))"),
    repeat(
        // name, width, type, default, min, max, help
        "repeat", 0, testsweeper::ParamType::Value, 1, 1, 1000,
        "times to repeat each test"),
    time(
        // name, width, precision, type, default, min, max, help
        "time(s)", 9, 4, testsweeper::ParamType::Output,
        testsweeper::no_data_flag, 0, 0, "time to solution"),
    gflops(
        // name, width, precision, type, default, min, max, help
        "gflops", 9, 4, testsweeper::ParamType::Output,
        testsweeper::no_data_flag, 0, 0, "gflop/s rate"),
    gbytes(
        // name, width, precision, type, default, min, max, help
        "gbyte/s", 11, 4, testsweeper::ParamType::Output,
        testsweeper::no_data_flag, 0, 0, "gbyte/s rate (bandwidth)"),
    ref_time(
        // name, width, precision, type, default, min, max, help
        "ref_time(s)", 11, 4, testsweeper::ParamType::Output,
        testsweeper::no_data_flag, 0, 0, "reference time to solution"),
    ref_gflops(
        // name, width, precision, type, default, min, max, help
        "ref_gflops",  11, 4, testsweeper::ParamType::Output,
        testsweeper::no_data_flag, 0, 0, "reference gflop/s rate"),
    ref_gbytes(
        // name, width, precision, type, default, min, max, help
        "ref_gbyte/s", 11, 4, testsweeper::ParamType::Output,
        testsweeper::no_data_flag, 0, 0, "reference gbyte/s rate (bandwidth)"),
    latms_cond(
        // name, width, precision, type, default, min, max, help
        "latms_cond", 0, 0, testsweeper::ParamType::Value, 1,
        -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
        "used in setting D as described for the mode parameter"),
    latms_dmax(
        // name, width, precision, type, default, min, max, help
        "latms_dmax", 0, 0, testsweeper::ParamType::Value, -1,
        -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
        "abs(dmax) is the maximum absolute eigenvalue/singular value"),
    accuracy(
        // name, width, precision, type, default, min, max, help
        "accuracy", 8, 0, testsweeper::ParamType::List, 1e-4, 0, 1,
        "accuracy threshold"),
    error(
        // name, width, precision, type, default, min, max, help
        "error", 11, 4, testsweeper::ParamType::Output,
        testsweeper::no_data_flag, 0, 0, "numerical error"),
    rk(
        // name, width, type, default, help
        "rk", 6, testsweeper::ParamType::Output, "NA",
        "numerical rank growth"),
    okay(
        // name, width, type, default (-1 means "no check"), min, max, help
        "status", 6, testsweeper::ParamType::Output, -1, 0, 0,
        "success indicator"),
    verbose(
        // name, width, type, default, min, max, help
        "verbose", 0, testsweeper::ParamType::Value, 0, 0, 10, "verbose level"),
    tol(
        // name, width, precision, type, default, min, max, help
        "tol", 0, 0, testsweeper::ParamType::Value, 3, 1, 1000,
        "tolerance (e.g., error < tol*accuracy to pass)")
{
    okay();
    error();
    time();
    ref_time();
    gflops();
    ref_gflops();

    check();
    repeat();
    verbose();
}

std::string version2str(int version) {
    char buf[1024];
    snprintf(buf, sizeof(buf), "%d.%02d.%02d",
             version / 10000, (version % 10000) / 100, version % 100);
    return std::string(buf);
}

int main(int argc, char* argv[]) {
    assert((sizeof(section_names) / sizeof(*section_names)) ==
           (Section::num_sections));

    int status = 0;
    try {
        printf("HCORE version %s, id %s\n", hcore::version(), hcore::id());
        printf("BLAS++ version %s, id %s\n",
               version2str(blas::blaspp_version()).c_str(),
               blas::blaspp_id());
        printf("LAPACK++ version %s, id %s\n",
               version2str(lapack::lapackpp_version()).c_str(),
               lapack::lapackpp_id());
        printf("TestSweeper version %s, id %s\n",
               version2str(testsweeper::version()).c_str(),
               testsweeper::id());

        printf("input: %s", argv[0]);
        for (int i = 1; i < argc; ++i) {
            std::string a(argv[i]);
            const char* wordchars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO"
                                    "PQRSTUVWXYZ0123456789_-=";
            if (a.find_first_not_of(wordchars) != std::string::npos)
                printf(" '%s'", argv[i]);
            else
                printf(" %s", argv[i]);
        }
        printf("\n");

        std::time_t t = std::time(nullptr);
        char current_time[100];
        std::strftime(current_time, sizeof(current_time), "%F %T",
                      std::localtime(&t));
        printf("%s.\n", current_time);

        if (argc < 2 || strcmp(argv[argc-1], "-h")     == 0
                     || strcmp(argv[argc-1], "--help") == 0) {
            testsweeper::usage(argc, argv, routines, section_names);
            throw testsweeper::QuitException();
        }

        const char* routine = argv[argc-1];
        testsweeper::test_func_ptr tester = find_tester(routine, routines);
        if (tester == nullptr) {
            testsweeper::usage(argc, argv, routines, section_names);
            throw std::runtime_error(std::string("routine ") + routine +
                                     std::string(" not found"));
        }

        Params p;
        p.routine = routine;
        tester(p, false);
        try {
            p.parse(routine, argc-2, argv+1);
        }
        catch (...) {
            p.help(routine);
            throw;
        }

        testsweeper::DataType last = p.datatype();
        p.header();
        do {
            if (p.datatype() != last) {
                last = p.datatype();
                printf("\n");
            }
            for (int i = 0; i < p.repeat(); ++i) {
                try {
                    tester(p, true);
                }
                catch (const std::exception& e) {
                    fprintf(stderr, "\n%s%s%s%s\n", testsweeper::ansi_bold,
                            testsweeper::ansi_red, e.what(),
                            testsweeper::ansi_normal);
                    p.okay() = false;
                }

                p.print();
                std::fflush(stdout);
                status += !p.okay();
                p.reset_output();
            }

            if (p.repeat() > 1)
                printf("\n");

        } while (p.next());

        if (status)
            printf("%d tests FAILED for %s.\n", status, routine);
        else
            printf("All tests passed for %s.\n", routine);
    }
    catch (const testsweeper::QuitException& e) {
    }
    catch (const std::exception& e) {
        fprintf(stderr, "\n%s%s%s%s\n", testsweeper::ansi_bold,
                testsweeper::ansi_red, e.what(), testsweeper::ansi_normal);
        status = -1;
    }

    return status;
}
