add_subdirectory(kernels)
add_subdirectory(api)
add_subdirectory(data-units)
add_subdirectory(operators)
add_subdirectory(helpers)

set(LIB_NAME ${PROJECT_NAME})

add_library(${LIB_NAME}
        STATIC
        ${SOURCES}
        )

target_link_libraries(${LIB_NAME} ${LIBS})
message("Libraries ${LIBS}")
if (USE_CUDA)
    message(" CUDA LIBRARIES  : " ${CUDA_LIBRARIES})
    message(" CUDA CUDA_CUBLAS_LIBRARIES  : " ${CUDA_CUBLAS_LIBRARIES})
    target_link_libraries(${LIB_NAME} ${LIBS} ${CUDA_LIBRARIES}
            -lcusolver -lcublas -lcublasLt
            )
endif ()
target_compile_definitions(${LIB_NAME} PUBLIC ${COMPILE_DEFINITIONS})
