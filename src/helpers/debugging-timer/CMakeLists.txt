if (HCOREPP_USE_TIMER)
    set(SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/ActiveTimer.cpp
            ${SOURCES}
            PARENT_SCOPE)
else ()
    set(SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/IdleTimer.cpp
            ${SOURCES}
            PARENT_SCOPE)
endif ()