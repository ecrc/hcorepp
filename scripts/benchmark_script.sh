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

PER_TILE_OPTIONS=(1 0)
acc="1e-1,1e-4,1e-8,1e-12"

for per_tile in ${PER_TILE_OPTIONS[@]}; do
  
  export HCOREPP_VERBOSE=ON
  TileCount=(1 2 4 8 12 16 20 24 28 32 36 40 44 48)

  cat /dev/null > benchmark_ts512_$per_tile.csv

  for tile_count in ${TileCount[@]}; do
        $1 $tile_count $acc 512 $per_tile >> benchmark_ts512_$per_tile.csv
        unset HCOREPP_VERBOSE
  done 
done
