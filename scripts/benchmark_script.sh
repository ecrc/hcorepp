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
TileCount=(1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36)

cat /dev/null > benchmark_ts1024_1.csv

for tile_count in ${TileCount[@]}; do
      $1 $tile_count $acc 1024 1 >> benchmark_ts1024_1.csv
      unset HCOREPP_VERBOSE
done
