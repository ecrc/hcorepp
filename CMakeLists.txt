# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
# Project settings.
project(hcorepp VERSION 1.0.0)

if (USE_CUDA)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
else ()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prerequisites)

add_subdirectory(src)

#option(BUILD_TESTS "Option to enable building tests" OFF)
#if (${BUILD_TESTS})
message(STATUS "Building Tests")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
include(CTest)
add_test(NAME hcorepp-tests
        COMMAND hcorepp-tests
        )
#endif ()

# search for BLAS library, if not already included
message("")
message("---------------------------------------- BLAS++")
message(STATUS "Checking for BLAS++")
if (NOT TARGET blaspp)
    find_package(blaspp QUIET)
    if (blaspp_FOUND)
        message("   Found BLAS++: ${blaspp_DIR}")
    elseif (EXISTS "${CMAKE_SOURCE_DIR}/blaspp/CMakeLists.txt")
        set(build_tests_save "${build_tests}")
        set(build_tests "false")

        add_subdirectory("blaspp")

        set(build_tests "${build_tests_save}")
        set(blaspp_DIR "${CMAKE_BINARY_DIR}/blaspp")
    else()
        # message(
        #     FATAL_ERROR
        #     "BLAS++ not found.
        #       HCORE requires BLAS++ to be installed first."
        # )
        set(build_tests_save "${build_tests}")
        set(build_tests "false")

        set(url "https://bitbucket.org/icl/blaspp")
        # set(tag "2020.08.00")
        # set(tag "2020.09.00")
        # set(tag "2020.10.00")
        # set(tag "2020.10.01")
        # set(tag "2020.10.02")
        # set(tag "2020.10.02")
        # set(tag "2021.04.00")
        set(tag "2021.04.01")
        message(STATUS "Fetching BLAS++ ${tag} from ${url}")
        include(FetchContent)
        FetchContent_Declare(
                blaspp GIT_REPOSITORY "${url}" GIT_TAG "${tag}")
        FetchContent_MakeAvailable(blaspp)

        set(build_tests "${build_tests_save}")
    endif()
else()
    message("   BLAS++ already included")
endif()
message(STATUS "BLAS++ done")
