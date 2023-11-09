# Add example to build list.
add_executable(hcorepp-trsm ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(hcorepp-trsm ${PROJECT_NAME})