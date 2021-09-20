// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_INTERNAL_HH
#define HCORE_INTERNAL_HH

#include <cstdint>

#include "hcore/compressed_tile.hh"

namespace hcore {
namespace internal {

template <typename T>
void rsvd(blas::Op transA, blas::Op transB,
          T beta, T const* AU, int64_t ldau,
                  T const* AV, int64_t ldav, int64_t Ark,
                  CompressedTile<T>& C,
          bool use_gemm=true, bool truncate_with_tol=false, int64_t fixed_rk=0);

} // namespace internal
} // namespace hcore

#endif // HCORE_INTERNAL_HH