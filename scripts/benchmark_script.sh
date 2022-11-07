#!/bin/bash
#
# Copyright (c) 2017-2022, King Abdullah University of Science and Technology
# ***************************************************************************
# *****      KAUST Extreme Computing Research Center Property           *****
# ***************************************************************************
#
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
#
if [[ $# -eq 0 ]] ; then
    echo 'This script needs a single argument that is the hcorepp-matrix binary to benchmark'
    exit 0
fi

acc="1e-1,1e-2,1e-4,1e-6,1e-8,1e-10"
  
export HCOREPP_VERBOSE=ON
TileCount=(1 2 4 8 12 16 20 24 28 32 36 40 44 48)

cat /dev/null > benchmark_ts512_0.csv

for tile_count in ${TileCount[@]}; do
      $1 $tile_count $acc 512 0 >> benchmark_ts512_0.csv
      unset HCOREPP_VERBOSE
done
