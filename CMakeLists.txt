# set minimum cmake version
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
# Project settings.
project(hcorepp VERSION 1.1.0 DESCRIPTION "BLAS for Tile Low-rank Matrix Algebra")
# Project Options.
option(USE_CUDA "Use Cuda, if available" false)
option(HCOREPP_BUILD_TESTS "Option to enable building tests" ON)
option(HCOREPP_BUILD_EXAMPLES "Option to enable building examples" ON)
option(HCOREPP_BUILD_DOCS "Build documentation in docs directory" ON)

message("CMAKE VERSION: ${CMAKE_VERSION}")
enable_language(CXX)
# Cmake Module Paths
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Toolchain selection
if (USE_CUDA)
    add_definitions(-DUSE_CUDA=TRUE)
    enable_language(CUDA)
    include(toolchains/CudaToolchain)
    set(BLA_VENDOR NVHPC)
else ()
    include(toolchains/GccToolchain)
endif ()

# Add all dependencies for HCorePP
include(ImportGFortran)
include(ImportOpenMP)
if (USE_CUDA)
    message("-- Build CUDA Support")
    include(ImportCuSolver)
    include(ImportBlasPP)
    unset(BLA_VENDOR)
    include(ImportLapack)
    set(LIBS
            LAPACK::LAPACK
            ${LIBS}
            )
else ()
    message("-- Build x86 Support")
    set(gpu_backend CACHE "none" FORCE)
    include(ImportBlasPP)
    include(ImportLapackPP)
endif ()

if(HCOREPP_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory("docs")
    else()
        message(STATUS "Doxygen NOT found, skipping it")
    endif()
endif()

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

# Add examples if enabled
if (${HCOREPP_BUILD_EXAMPLES})
    message(STATUS "Building Examples")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif ()

# Installation actions
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)
## Install cmake find package.
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" COMPATIBILITY ExactVersion)
install(
        FILES
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION lib/cmake/${PROJECT_NAME}
)

configure_file(${PROJECT_NAME}Config.cmake.in
        ${PROJECT_NAME}Config.cmake @ONLY)

install(
        FILES
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
        DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
        DESTINATION lib/cmake/${PROJECT_NAME}/Modules
)

## Generate pkg-config file
configure_file(package.pc.in
        lib/pkgconfig/${PROJECT_NAME}.pc @ONLY)
install(
        FILES
        "${PROJECT_BINARY_DIR}/lib/pkgconfig/${PROJECT_NAME}.pc"
        DESTINATION lib/pkgconfig/
)


##################
# Release source #
##################
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HCore++ library ")
set(CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_VENDOR "KAUST")
set(CPACK_PACKAGE_CONTACT "hatem.ltaief@kaust.edu.sa")
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
set(CPACK_SOURCE_IGNORE_FILES "bin;.git;.gitmodules;Jenkinsfile")
include(CPack)
