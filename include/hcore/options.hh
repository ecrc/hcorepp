// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_OPTIONS_HH
#define HCORE_OPTIONS_HH

#include <cstdint>
#include <map>

namespace hcore {

enum class Option : char {
    UseGEMM  = 'U',
    FixedRank = 'F',
    TruncateWithTol = 'T',
};

struct OptionValue {
public:
    OptionValue() {}

    OptionValue(int i) : i_(i) {}

    OptionValue(bool i) : i_(i) {}

    OptionValue(int64_t i) : i_(i) {}

    int64_t i_;
};

using Options = std::map<Option, OptionValue>;

template <typename T>
T get_option(Options opts, Option option, T defval)
{
    if (opts.empty()) return defval; // quick return

    T retval;
    auto search = opts.find(option);
    if (search != opts.end())
        retval = T(search->second.i_);
    else
        retval = defval;

    return retval;
}

}  // namespace hcore

#endif  // HCORE_OPTIONS_HH