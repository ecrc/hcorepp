// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_EXCEPTION_HH
#define HCORE_EXCEPTION_HH

#include <string>
#include <stdexcept>

#define hcore_throw_std_invalid_argument_if(condition) \
    do { \
        if (condition) { \
            const std::string& what_arg = \
                "HCORE throws an std::invalid_argument exception: (" \
                + std::string(#condition) + "), function " \
                + __func__ + ", file " + __FILE__ + ", line " \
                + std::to_string(__LINE__) + "."; \
            throw std::invalid_argument(what_arg); \
        } \
    } while(0)

#endif // HCORE_EXCEPTION_HH
