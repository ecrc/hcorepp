#!/bin/bash
#
# Copyright (c) 2017-2022, King Abdullah University of Science and Technology
# ***************************************************************************
# *****      KAUST Extreme Computing and Research Center Property       *****
# ***************************************************************************
#
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
#
if [[ $# -eq 0 ]] ; then
    echo 'This script needs a single argument that is the hcorepp-matrix binary to benchmark'
    exit 0
fi

export HCOREPP_VERBOSE=ON
TileCount=(1 2 4 8 16 32 64 128)
acc="1e-1,1e-2,1e-4,1e-8"

cat /dev/null > benchmark_ts512.csv

for tile_count in ${TileCount[@]}; do
      $1 $tile_count $acc 512 >> benchmark_ts512.csv
      unset HCOREPP_VERBOSE
done


export HCOREPP_VERBOSE=ON

cat /dev/null > benchmark_t1.csv

Prop=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
for tile_size in ${Prop[@]}; do
      $1 1 "$acc" $tile_size >> benchmark_t1.csv
      unset HCOREPP_VERBOSE
done


export HCOREPP_VERBOSE=ON
TileCount=(1 2 4 8 16 32)

cat /dev/null > benchmark_ts2048.csv

for tile_count in ${TileCount[@]}; do
      $1 $tile_count $acc 2048 >> benchmark_ts2048.csv
      unset HCOREPP_VERBOSE
done
