add_subdirectory(api)
add_subdirectory(data-units)
add_subdirectory(operators)

set(LIB_NAME ${PROJECT_NAME})

add_library(${LIB_NAME}
        STATIC
        ${SOURCES}
        )
target_link_libraries(${LIB_NAME} ${LIBS})
target_compile_definitions(${LIB_NAME} PUBLIC ${COMPILE_DEFINITIONS})