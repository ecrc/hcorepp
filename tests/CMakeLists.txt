set(HCOREPP_TESTFILES ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp helpers/include/hcorepp/test-helpers/lapack_wrappers.hpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/helpers/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/helpers/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/helpers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data-units)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/api)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/operators)

enable_testing()

add_executable(hcorepp-tests ${HCOREPP_TESTFILES})
target_link_libraries(hcorepp-tests ${PROJECT_NAME})