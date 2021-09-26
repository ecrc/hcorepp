# Copyright (c) 2017-2021,
# King Abdullah University of Science and Technology (KAUST).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

# CXX compiler must match the one used to compiler HCORE
# set it in your environment

cmake_minimum_required(VERSION 3.8)

project(hcore_example LANGUAGES CXX)

add_executable(example_gemm_ccc example_gemm_ccc.cc)

find_package(hcore REQUIRED)

target_link_libraries(example_gemm_ccc hcore)
