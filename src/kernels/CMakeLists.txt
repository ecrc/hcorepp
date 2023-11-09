if (${USE_CUDA})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda)
elseif (${USE_SYCL})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sycl)
else ()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/omp)

endif ()

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ContextManager.cpp
        ${SOURCES}
        PARENT_SCOPE
        )
