# Copyright (c) 2017-2021, King Abdullah University of Science and Technology
# (KAUST). All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

# CMake script for HCORE, https://github.com/ecrc/hcorepp
# Requires:
#     BLAS++, http://bitbucket.org/icl/blaspp
#     LAPACK++, http://bitbucket.org/icl/lapackpp
# Tests requires:
#     TestSweeper, http://bitbucket.org/icl/testsweeper

cmake_minimum_required(VERSION 3.15)
# 3.1 target_compile_features
# 3.8 target_compile_features(cxx_std_11)
# 3.14 install(LIBRARY DESTINATION lib) default
# 3.15 $<$COMPILE_LANG_AND_ID
# 3.15 message DEBUG, string REPEAT
# 3.15 cmake_minimum_required(VERSION 3.15) of BLAS++, LAPACK++, and TestSweeper 

project(
    hcore
    VERSION 1.0.0 # major.minor.patch
    LANGUAGES CXX
)

# when built as a sub-project, add a namespace to make targets unique.
# For instance, `make tester` becomes `make hcore_tester`.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(hcore_is_project true)
    set(hcore_ "")
else()
    set(hcore_is_project false)
    set(hcore_ "hcore_")
endif()

if (hcore_is_project)
    set(log "" CACHE STRING "Shorthand for CMAKE_MESSAGE_LOG_LEVEL")
    set_property(CACHE log
        PROPERTY STRINGS FATAL_ERROR SEND_ERROR WARNING AUTHOR_WARNING
        DEPRECATION NOTICE STATUS VERBOSE DEBUG TRACE)
    if (log)
        set(CMAKE_MESSAGE_LOG_LEVEL "${log}")
    endif()
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" true)
option(use_openmp "Use OpenMP, if available" true)
option(build_tests "Build hcore test suite" "${hcore_is_project}")
option(color "Use ANSI color output" true)

# default prefix=/opt/hcore
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND hcore_is_project)
    set(prefix "/opt/hcore" CACHE PATH "Shorthand for CMAKE_INSTALL_PREFIX")
    set(CMAKE_INSTALL_PREFIX "${prefix}"
        CACHE PATH "Install path prefix, prepended onto install directories."
        FORCE
    )
    message(STATUS "Setting CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
    # append the new CMAKE_INSTALL_PREFIX, since CMake appended the old value.
    # this helps to find TestSweeper
    list(APPEND CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_INSTALL_PREFIX})
else()
    message(STATUS "Using CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
endif()

# GNU filesystem conventions
include(GNUInstallDirs)

# the RPATH to be used when installing, but only if it's not a system directory
list(
    FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
    list(APPEND
        CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

# provide menu of options
set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS None Debug Release RelWithDebInfo MinSizeRel)

# HCORE defaults to single-threaded BLAS
set(
    blas_threaded "false"
    CACHE STRING "Multi-threaded BLAS? (Passed to BLAS++.)" )
set_property(
    CACHE blas_threaded PROPERTY STRINGS "auto" "true" "false" )

message(DEBUG "Settings:
CMAKE_INSTALL_PREFIX   = ${CMAKE_INSTALL_PREFIX}
CMAKE_BUILD_TYPE       = ${CMAKE_BUILD_TYPE}
BUILD_SHARED_LIBS      = ${BUILD_SHARED_LIBS}
build_tests            = ${build_tests}
color                  = ${color}
hcore_is_project       = ${hcore_is_project}
hcore_                 = ${hcore_}
")

# enforce out-of-source build
string(TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir)
string(TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir)
if ("${source_dir}" STREQUAL "${binary_dir}")
    message(
        FATAL_ERROR
        "Compiling HCORE with CMake requires an out-of-source build. To proceed:
        # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
        rm -rf CMakeCache.txt CMakeFiles/
        mkdir build
        cd build
        cmake ..
        make"
    )
endif()

# build library
file(
    GLOB libhcore_src
    CONFIGURE_DEPENDS # glob at build time
    src/*.cc
    src/internal/*.cc
)
message(DEBUG "libhcore_src = ${libhcore_src}")

add_library(
    hcore
    ${libhcore_src}
)

# include directory
# during build it's {source}/include; after install it's {prefix}/include
target_include_directories(
    hcore
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# OpenMP support.
if (NOT use_openmp)
    message(STATUS "User has requested to NOT use OpenMP")
else()
    find_package(OpenMP)
endif()
if (OpenMP_CXX_FOUND)
    target_link_libraries(hcore PUBLIC "OpenMP::OpenMP_CXX")
else()
    message(STATUS "No OpenMP support in HCORE")
endif()

# get git commit id
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE hcore_id)
    string(STRIP "${hcore_id}" hcore_id)
    message(STATUS "hcore_id = ${hcore_id}")
    target_compile_definitions(hcore PRIVATE HCORE_ID="${hcore_id}")
endif()

# use and export -std=c++11; don't allow -std=gnu++11 extensions.
target_compile_features(hcore PUBLIC cxx_std_11)
set_target_properties(hcore PROPERTIES CXX_EXTENSIONS OFF)

# forcing specific CXX standard causing issues with NEC compiler
# # use and export -std=c++17; don't allow -std=gnu++17 extensions
# target_compile_features(hcore PUBLIC cxx_std_17)
# set_target_properties(hcore
#     PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED true CXX_EXTENSIONS false)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    # conditionally add -Wall. See CMake tutorial
    set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>")
    target_compile_options(
        hcore PRIVATE "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall>>" )
endif()

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

# search for LAPACK library, if not already included
message("")
message("---------------------------------------- LAPACK++")
message(STATUS "Checking for LAPACK++")
if (NOT TARGET lapackpp)
    find_package(lapackpp QUIET)
    if (lapackpp_FOUND)
        message("   Found LAPACK++: ${lapackpp_DIR}")
    elseif (EXISTS "${CMAKE_SOURCE_DIR}/lapackpp/CMakeLists.txt")
        set(build_tests_save "${build_tests}")
        set(build_tests "false")

        add_subdirectory("lapackpp")

        set(build_tests "${build_tests_save}")
        set(lapackpp_DIR "${CMAKE_BINARY_DIR}/lapackpp")
    else()
        # message(
        #     FATAL_ERROR
        #     "LAPACK++ not found.
        #       HCORE requires LAPACK++ to be installed first."
        # )
        set(build_tests_save "${build_tests}")
        set(build_tests "false")

        set(url "https://bitbucket.org/icl/lapackpp")
        # set(tag "2020.09.00")
        # set(tag "2020.10.00")
        # set(tag "2020.10.01")
        # set(tag "2020.10.02")
        set(tag "2021.04.00")
        message(STATUS "Fetching LAPACK++ ${tag} from ${url}")
        include(FetchContent)
        FetchContent_Declare(
            lapackpp GIT_REPOSITORY "${url}" GIT_TAG "${tag}")
        FetchContent_MakeAvailable(lapackpp)

        set(build_tests "${build_tests_save}")
    endif()
else()
    message("   LAPACK++ already included")
endif()
message(STATUS "LAPACK++ done")

target_link_libraries(hcore PUBLIC blaspp lapackpp)

# if ("${blaspp_defines}" MATCHES "HAVE_MKL")
#     target_compile_definitions(hcore PUBLIC "-DHCORE_")
# endif()

if ("${lapackpp_defines}" MATCHES "LAPACK_ILP64")
    target_compile_definitions(hcore PUBLIC "-DHCORE_HAVE_LAPACK_WITH_ILP64")
endif()

if (color)
    string(ASCII 27 Esc)
    set(red   "${Esc}[31m")
    set(blue  "${Esc}[34m")
    set(plain "${Esc}[39m")
endif()

if (build_tests)
    message("")
    message("---------------------------------------- LAPACK MATGEN")
    message(STATUS "Checking for LAPACK LATMS")

    message(DEBUG "lapackpp_libraries: ${lapackpp_libraries}")

    # cache lapackpp_libraries variable for future usage
    set(lapackpp_libraries_ ${lapackpp_libraries})
    message(DEBUG "lapackpp_libraries_: ${lapackpp_libraries_}")

    # remove blaspp library from the linking path because the blaspp library
    # doesn't exist yet.
    list(REMOVE_ITEM lapackpp_libraries_ blaspp)
    message(DEBUG "lapackpp_libraries_: ${lapackpp_libraries_}")
    message(DEBUG "LAPACK_LATMS_LIBRARIES: ${LAPACK_LATMS_LIBRARIES}")

    try_run(run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_latms.cc"
        LINK_LIBRARIES
            ${blaspp_libraries} ${lapackpp_libraries_} ${LAPACK_LATMS_LIBRARIES}
        COMPILE_DEFINITIONS
            ${blaspp_defines} ${lapackpp_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )

    message(DEBUG "lapack_latms.cc compile result: '${compile_result}'")
    message(DEBUG "lapack_latms.cc run result: '${run_result}'")
    message(TRACE "lapack_latms.cc compile_output: '''\n${compile_output}'''")
    message(TRACE "lapack_latms.cc run_output: '''\n${run_output}'''")

    if (compile_result AND "${run_output}" MATCHES "ok")
        set(lapack_latms_found_ true CACHE INTERNAL "")
    endif()

    if (lapack_latms_found_)
        if (NOT LAPACK_LATMS_LIBRARIES)
            message("   ${blue}Found LAPACK LATMS in BLAS/LAPACK library${plain}")
        else()
            message("   ${blue}Found LAPACK LATMS in ${LAPACK_LATMS_LIBRARIES}${plain}")
        endif()
        message("   ${blue}Tester will use LAPACK LATMS for the random "
                "matrix generation${plain}")
        target_compile_definitions(hcore PUBLIC "-DHCORE_HAVE_LAPACK_WITH_TMG")
    else()
        message("   ${red}LAPACK LATMS not found${plain}")
        message("   ${red}Tester will use LAPACK LARNV for the random "
                "matrix generation${plain}")
    endif()

    message(STATUS "LAPACK MATGEN done")
    add_subdirectory(test)
else()
    message("")
endif()

# add 'make lib' target
if (hcore_is_project)
    add_custom_target(lib DEPENDS hcore)
endif()

# install rules
# GNU filesystem conventions
# include(GNUInstallDirs)
set(install_configdir "${CMAKE_INSTALL_LIBDIR}/hcore")

# install library and add to <package>Targets.cmake
install(
    TARGETS hcore
    EXPORT hcoreTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    # RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# install header files
install(
    # / copies contents, not directory itself
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING REGEX "\\.(h|hh)$"
)

# install <package>Targets.cmake
install(
    EXPORT hcoreTargets
    DESTINATION "${install_configdir}"
)

# also export <package>Targets.cmake in build directory
export(
    EXPORT hcoreTargets
    FILE "hcoreTargets.cmake"
)

# install <package>Config.cmake and <package>ConfigVersion.cmake,
# to enable find_package(<package>).
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "hcoreConfig.cmake.in"
    "hcoreConfig.cmake"
    INSTALL_DESTINATION "${install_configdir}"
)
write_basic_package_version_file(
    "hcoreConfigVersion.cmake"
    VERSION "${hcore_VERSION}"
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/hcoreConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/hcoreConfigVersion.cmake"
    DESTINATION "${install_configdir}"
)
