set(HCOREPP_TESTFILES ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data-units)

enable_testing()

add_executable(hcorepp-tests ${HCOREPP_TESTFILES})
target_link_libraries(hcorepp-tests ${PROJECT_NAME})