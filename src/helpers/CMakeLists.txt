add_subdirectory(debugging-timer)

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/MatrixHelpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RawMatrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TileMatrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/generators/LatmsGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/generators/TileLatmsGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/generators/RandomGenerator.cpp
        ${SOURCES}
        PARENT_SCOPE
        )