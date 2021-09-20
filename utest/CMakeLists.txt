# Copyright (c) 2017-2021,
# King Abdullah University of Science and Technology (KAUST).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

# search for TestSweeper library, if not already included
message("")
message("---------------------------------------- TestSweeper")
message(STATUS "Checking for TestSweeper")
if (NOT TARGET testsweeper)
    find_package(testsweeper QUIET)
    if (testsweeper_FOUND)
        message("   Found TestSweeper: ${testsweeper_DIR}")
    elseif (EXISTS "${CMAKE_SOURCE_DIR}/testsweeper/CMakeLists.txt")
        set(build_tests_save "${build_tests}")
        set(build_tests "false")

        add_subdirectory(
            "${CMAKE_SOURCE_DIR}/testsweeper"
            "${CMAKE_BINARY_DIR}/testsweeper"
        )

        set(build_tests "${build_tests_save}")
        set(testsweeper_DIR "${CMAKE_BINARY_DIR}/testsweeper")
    else()
        # message(
        #     FATAL_ERROR
        #     "TestSweeper not found.
        #       HCORE requires TestSweeper to be installed first."
        # )
        set(build_tests_save "${build_tests}")
        set(build_tests "false")

        set(url "https://bitbucket.org/icl/testsweeper")
        # set(tag "2020.03.00")
        # set(tag "2020.06.00")
        # set(tag "2020.09.00")
        set(tag "2021.04.00")
        message(STATUS "Fetching TestSweeper ${tag} from ${url}")
        include(FetchContent)
        FetchContent_Declare(
            testsweeper GIT_REPOSITORY "${url}" GIT_TAG "${tag}")
        FetchContent_MakeAvailable(testsweeper)

        set(build_tests "${build_tests_save}")
    endif()
else()
    message("   TestSweeper already included")
endif()
message(STATUS "TestSweeper done")
message("")

# ------------------------------------------------------------------------------
# test/unit/tile.cc
# file(
#     GLOB unit_test_tile_src
#     CONFIGURE_DEPENDS
#     tile.cc
#     unit_test.cc
# )
#
# set(unit_test_tile "${hcore_}unit_test_tile")
# add_executable(
#     ${unit_test_tile}
#     ${unit_test_tile_src}
# )
#
# # C++11 is inherited from hcore, but disabling extensions is not
# set_target_properties(${unit_test_tile} PROPERTIES CXX_EXTENSIONS false)
#
# target_link_libraries(
#     ${unit_test_tile}
#     testsweeper
#     hcore
# )

# ------------------------------------------------------------------------------
# test/unit/gemm.cc
file(
    GLOB utest_gemm_src
    CONFIGURE_DEPENDS
    utest_gemm.cc
    test.cc
)

set(utest_gemm "${hcore_}utest_gemm")
add_executable(
    ${utest_gemm}
    ${utest_gemm_src}
)

# C++11 is inherited from hcore, but disabling extensions is not
set_target_properties(${utest_gemm} PROPERTIES CXX_EXTENSIONS false)

target_link_libraries(
    ${utest_gemm}
    testsweeper
    hcore
)

# # copy run_tests script to build directory
# add_custom_command(
#     TARGET ${tester} POST_BUILD
#     COMMAND
#         cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
#            ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
# )
#
# if (hcore_is_project)
#     # add 'make check' target
#     add_custom_target(
#         "check"
#         COMMAND python run_tests.py --quick
#         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#     )
# endif()
