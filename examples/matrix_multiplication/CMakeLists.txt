# Add example to build list.
add_executable(omp-hcorepp-matrix ${CMAKE_CURRENT_SOURCE_DIR}/omp_main.cpp)
target_link_libraries(omp-hcorepp-matrix ${PROJECT_NAME})

add_executable(par-hcorepp-matrix ${CMAKE_CURRENT_SOURCE_DIR}/par_main.cpp)
target_link_libraries(par-hcorepp-matrix ${PROJECT_NAME})

add_executable(fixed-par-hcorepp-matrix ${CMAKE_CURRENT_SOURCE_DIR}/par_fixed_rank_main.cpp)
target_link_libraries(fixed-par-hcorepp-matrix ${PROJECT_NAME})

add_executable(fixed-par-streams-hcorepp-matrix ${CMAKE_CURRENT_SOURCE_DIR}/par_fixed_rank_streams_main.cpp)
target_link_libraries(fixed-par-streams-hcorepp-matrix ${PROJECT_NAME})

add_executable(seq-hcorepp-matrix ${CMAKE_CURRENT_SOURCE_DIR}/seq_main.cpp)
target_link_libraries(seq-hcorepp-matrix ${PROJECT_NAME})