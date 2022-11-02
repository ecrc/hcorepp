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
acc="1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8"

cat /dev/null > benchmark_512.csv

for tile_count in ${TileCount[@]}; do
      $1 $tile_count $acc $tile_size 512 >> benchmark_512.csv
      unset HCOREPP_VERBOSE
done


export HCOREPP_VERBOSE=ON

cat /dev/null > benchmark_1024.csv

TileCount=(1 2 4 8 16 32 64)
for tile_count in ${TileCount[@]}; do
      $1 $tile_count "$acc" $tile_size 1024 >> benchmark_1024.csv
      unset HCOREPP_VERBOSE
done
