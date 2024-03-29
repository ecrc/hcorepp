# search for LAPACK library, if not already included
message("")
message("---------------------------------------- LAPACK")
message(STATUS "Checking for LAPACK")

include(macros/BuildDependency)

if (NOT TARGET LAPACK)
    find_package(LAPACK REQUIRED)
    if (LAPACK_FOUND)
        message("   Found LAPACK: ${LAPACK_LIBRARIES}")
    else ()
        set(build_tests_save "${build_tests}")
        set(build_tests "false")
        BuildDependency(blas "https://github.com/xianyi/OpenBLAS" "v0.3.21")
        set(build_tests "${build_tests_save}")
        find_package(LAPACK REQUIRED)
    endif ()
else ()
    message("   LAPACK already included")
endif ()

message(STATUS "LAPACK done")