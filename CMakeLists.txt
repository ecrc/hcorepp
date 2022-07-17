# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
# Project settings.
project(hcorepp VERSION 1.0.0)
# Project Options.
option(USE_CUDA "Use Cuda, if available" false)
option(HCOREPP_BUILD_TESTS "Option to enable building tests" OFF)

# Cmake Module Paths
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Toolchain selection
if (USE_CUDA)
    include(toolchains/CudaToolchain)
else ()
    include(toolchains/GccToolchain)
endif ()

# Add all dependencies for HCorePP
include(ImportOpenMP)
include(ImportBlasPP)
include(ImportLapackPP)

# Include directories for HCorePP
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prerequisites)

# Add HCore library
add_subdirectory(src)

# Add tests if enabled
if (${HCOREPP_BUILD_TESTS})
    message(STATUS "Building Tests")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
    include(CTest)
    add_test(NAME hcorepp-tests
            COMMAND hcorepp-tests
            )
endif ()
