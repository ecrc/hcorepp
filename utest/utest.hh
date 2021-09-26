// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

namespace hcore {
namespace utest {

typedef void test_func_ptr(void);

void run(test_func_ptr* func, const char* name);
int main();

void launch(); ///> To be implemented by user -- called by unit::test::main()

} // namespace test
} // namespace hcore
