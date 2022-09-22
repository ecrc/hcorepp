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
        include(ImportLapack)
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
        include(ImportLapack)
#        set(BLA_VENDOR NVHPC)
#        set(LAPACK_LIBRARIES='-lopenblas')
#        find_package(LAPACK QUIET)

        set(url "https://bitbucket.org/icl/lapackpp")
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

# Add to linking libs.
set(LIBS
        lapackpp
        ${LIBS}
        )

# Add definition indicating version.
if ("${lapackpp_defines}" MATCHES "LAPACK_ILP64")
    set(COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS} -DHCORE_HAVE_LAPACK_WITH_ILP64")
endif()

message(STATUS "LAPACK++ done")