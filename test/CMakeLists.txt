# Copyright (c) 2017,
# King Abdullah University of Science and Technology (KAUST)
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

# todo
# can't build testers if CBLAS, LAPACK, LAPACKE, or TestSweeper are not found
# if (NOT blaspp_cblas_found)
#     message(WARNING "CBLAS not found; tester cannot be built.")
#     return()
# endif()
# if (NOT lapack_found_)
#     message(WARNING "LAPACK not found; tester cannot be built.")
#     return()
# endif()
# if (NOT lapacke_found)
#     message(WARNING "LAPACKE not found; tester cannot be built.")
#     return()
# endif()

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
        set(tag "2020.09.00")
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

file(
    GLOB tester_src
    CONFIGURE_DEPENDS
    *.cc
)

set(tester "${hcore_}tester")
add_executable(
    ${tester}
    ${tester_src}
)

# C++17 is inherited from hcore, but disabling extensions is not
set_target_properties(${tester} PROPERTIES CXX_EXTENSIONS false)

target_link_libraries(
    ${tester}
    testsweeper
    hcore
#     ${blaspp_cblas_libraries} # todo
#     ${lapack_libraries_} # todo
#     ${lapacke_libraries} # todo
)

# todo
# target_include_directories(
#     ${tester}
#     PRIVATE
#         "${blaspp_cblas_include}"
#         "${lapacke_include}"
# )

# copy run_tests script to build directory
add_custom_command(
    TARGET ${tester} POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)

if (hcore_is_project)
    # add 'make check' target
    add_custom_target(
        "check"
        COMMAND
            python run_tests.py --quick
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
