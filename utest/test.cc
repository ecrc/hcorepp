// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <stdio.h>
#include <string>

#include "testsweeper.hh"

#include "test.hh"

namespace hcore {
namespace test {
namespace unit {

static int total_ = 0;
static int pass_  = 0;
static int fail_  = 0;
static int skip_  = 0;

void run(test_func_ptr* func, const char* fname) {
    printf("%-60s", fname);
    fflush(stdout);
    ++total_;

    // try {
        func();
        printf("%spass%s\n", testsweeper::ansi_blue, testsweeper::ansi_normal);
        ++pass_;
    // }
    // todo
    // catch (SkipException& e) {
    //     printf("%sskipped: %s%s%s%s\n", testsweeper::ansi_magenta,
    //            testsweeper::ansi_normal, testsweeper::ansi_gray, e.what(),
    //            testsweeper::ansi_normal);
    //     --total_;
    //     ++skip_;
    // }
    // catch (AssertError& e) {
    //     printf("%s%sFAILED:%s\n\t%s%s%s\n", testsweeper::ansi_bold,
    //            testsweeper::ansi_red, testsweeper::ansi_normal,
    //            testsweeper::ansi_gray, e.what(), testsweeper::ansi_normal);
    //     ++fail_;
    // }
    // catch (std::exception& e) {
    //     AssertError err("unexpected exception: " + std::string(e.what()),
    //                     __FILE__, __LINE__);
    //     printf("%s%sFAILED:%s\n\t%s%s%s\n", testsweeper::ansi_bold,
    //            testsweeper::ansi_red, testsweeper::ansi_normal,
    //            testsweeper::ansi_gray, err.what(), testsweeper::ansi_normal);
    //     ++fail_;
    // }
    // catch (...) {
    //     AssertError err("unexpected exception: (unknown type)",
    //                     __FILE__, __LINE__);
    //     printf("%s%sFAILED:%s\n\t%s%s%s\n", testsweeper::ansi_bold,
    //            testsweeper::ansi_red, testsweeper::ansi_normal,
    //            testsweeper::ansi_gray, err.what(), testsweeper::ansi_normal);
    //     ++fail_;
    // }
}

int main() {
    launch();

    if (pass_ == total_) {
        printf("\n%spassed all tests (%d of %d)%s\n", testsweeper::ansi_blue,
               pass_, total_, testsweeper::ansi_normal);

        if (skip_ > 0) {
            printf("%sskipped %d tests%s\n", testsweeper::ansi_magenta, skip_,
                   testsweeper::ansi_normal);
        }

        return 0;
    }
    else {
        printf("\n%spassed:  %3d of %3d tests%s\n"
               "%s%sfailed:  %3d of %3d tests%s\n", testsweeper::ansi_blue,
               pass_, total_, testsweeper::ansi_normal, testsweeper::ansi_bold,
               testsweeper::ansi_red, fail_, total_, testsweeper::ansi_normal);

        if (skip_ > 0) {
            printf("%sskipped: %3d tests%s\n", testsweeper::ansi_magenta, skip_,
                   testsweeper::ansi_normal);
        }

        return -1;
    }
}

} // namespace hcore
} // namespace test
} // namespace unit