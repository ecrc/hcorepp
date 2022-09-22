if (${USE_CUDA})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda)
else ()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/omp)
endif ()

set(SOURCES
        ${SOURCES}
        PARENT_SCOPE
        )
