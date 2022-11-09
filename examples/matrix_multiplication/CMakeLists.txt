# Add example to build list.
add_executable(hcorepp-matrix ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(hcorepp-matrix ${PROJECT_NAME})