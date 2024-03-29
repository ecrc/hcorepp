#!/bin/bash
#
# @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                     All rights reserved.
#
if [[ $# -eq 0 ]] ; then
    echo 'This script needs a single argument that is the hcorepp-matrix binary to benchmark'
    exit 0
fi

acc="1e-4,1e-6,1e-8,1e-10"

export HCOREPP_VERBOSE=ON
TileCount=(1 4 8 12 16 20)

cat /dev/null > benchmark_ts1024_1.csv

for tile_count in ${TileCount[@]}; do
      $1 $tile_count $acc 1024 1 >> benchmark_ts1024_1.csv
      unset HCOREPP_VERBOSE
done
