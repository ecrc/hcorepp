add_executable(hcorepp-gemm-example ${CMAKE_CURRENT_SOURCE_DIR}/HCoreppGemmExample.cpp)
target_link_libraries(hcorepp-gemm-example ${PROJECT_NAME})

add_executable(hcorepp-2d-array-gemm-example ${CMAKE_CURRENT_SOURCE_DIR}/HCorepp2dArrayExample.cpp)
target_link_libraries(hcorepp-2d-array-gemm-example ${PROJECT_NAME})