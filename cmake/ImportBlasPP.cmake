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
        include(ImportBlas)
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
        include(ImportBlas)
        set(url "https://bitbucket.org/icl/blaspp")
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
set(LIBS
        blaspp
        ${LIBS}
        )
message(STATUS "BLAS++ done")