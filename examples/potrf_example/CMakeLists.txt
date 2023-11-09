# Add example to build list.
add_executable(hcorepp-potrf ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(hcorepp-potrf ${PROJECT_NAME})