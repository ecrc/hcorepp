# Add OpenMP if requested.
option(USE_OPENMP "Use OpenMP, if available" true)
if (NOT USE_OPENMP)
    message(STATUS "User has requested to NOT use OpenMP")
else()
    find_package(OpenMP)
    set(LIBS
            OpenMP::OpenMP_CXX
            ${LIBS}
            )
endif()