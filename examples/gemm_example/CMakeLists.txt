# Add example to build list.
add_executable(hcorepp-gemm ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(hcorepp-gemm ${PROJECT_NAME})