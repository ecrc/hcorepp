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
acc="1e-1,1e-4,1e-8"

for per_tile in ${PER_TILE_OPTIONS[@]}; do
  export HCOREPP_VERBOSE=ON
  TileCount=(10 25 40 55 70 85 100 115 130 145 160 175 190 205)


  cat /dev/null > benchmark_ts128_$per_tile.csv

  for tile_count in ${TileCount[@]}; do
        $1 $tile_count $acc 128 $per_tile >> benchmark_ts128_$per_tile.csv
        unset HCOREPP_VERBOSE
  done


  export HCOREPP_VERBOSE=ON
  TileCount=(1 2 4 8 16 32 64)

  cat /dev/null > benchmark_ts512_$per_tile.csv

  for tile_count in ${TileCount[@]}; do
        $1 $tile_count $acc 512 $per_tile >> benchmark_ts512_$per_tile.csv
        unset HCOREPP_VERBOSE
  done


  export HCOREPP_VERBOSE=ON

  cat /dev/null > benchmark_t1_$per_tile.csv

  Prop=(128 256 512 1024 2048 4096 8192 16384 32768)
  for tile_size in ${Prop[@]}; do
        $1 1 "$acc" $tile_size $per_tile >> benchmark_t1_$per_tile.csv
        unset HCOREPP_VERBOSE
  done


  export HCOREPP_VERBOSE=ON
  TileCount=(1 2 3 4 6 8 12 16)

  cat /dev/null > benchmark_ts2048_$per_tile.csv

  for tile_count in ${TileCount[@]}; do
        $1 $tile_count $acc 2048 $per_tile >> benchmark_ts2048_$per_tile.csv
        unset HCOREPP_VERBOSE
  done
done
