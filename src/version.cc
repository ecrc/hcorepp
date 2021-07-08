// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"

#ifndef HCORE_VERSION
#define HCORE_VERSION "1.0.0" // major.minor.patch
#endif

#ifndef HCORE_ID
#define HCORE_ID "unknown"
#endif

namespace hcore {

/// @return HCORE version.
const char* version() { return HCORE_VERSION; }

/// @return HCORE git commit hash ID.
const char* id() { return HCORE_ID; }

}  // namespace hcore
