# Add example to build list.
add_executable(hcorepp-syrk ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(hcorepp-syrk ${PROJECT_NAME})