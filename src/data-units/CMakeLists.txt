if(${USE_CUDA} OR ${USE_SYCL})
    set(MemoryHandler ${CMAKE_CURRENT_SOURCE_DIR}/memory-handlers/pool/MemoryHandler.cpp)
else()
    set(MemoryHandler ${CMAKE_CURRENT_SOURCE_DIR}/memory-handlers/on-demand/MemoryHandler.cpp)
endif()


set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/DataHolder.cpp
        ${MemoryHandler}
        ${SOURCES}
        PARENT_SCOPE
        )
