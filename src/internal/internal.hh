// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_INTERNAL_INTERNAL_HH
#define HCORE_INTERNAL_INTERNAL_HH

#include "hcore/tile/compressed.hh"

#include <cstdint>

namespace hcore {
namespace internal {

template <typename T>
void reduced_svd(
    T beta, T const* AU, T const* AV, int64_t ldau, int64_t Ark,
    CompressedTile<T>& C, bool use_trmm=false, bool use_ungqr=true,
    bool truncated_svd=false, int64_t fixed_rk=0);

} // namespace internal
} // namespace slate

#endif // HCORE_INTERNAL_INTERNAL_HH
