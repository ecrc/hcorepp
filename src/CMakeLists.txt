add_subdirectory(api)
add_subdirectory(data-units)
add_subdirectory(operators)

add_library(HCOREPP
        STATIC
        ${SOURCES}
        )