IF (NOT TARGET Catch2)
    find_package(Catch2 QUIET)

    # If Catch2 is not found, fetch and build it
    if (NOT Catch2_FOUND)
        message(STATUS "${Red}Couldn't find catch2 pre-installed, will begin fetching it v3.3.2${ColourReset}")
        include(FetchContent)
        set(FETCHCONTENT_QUIET OFF)
        FetchContent_Declare(
                Catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG v3.3.2 # Replace with the version of Catch2 you want to use for v3
                GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Catch2)
    else ()
        message(STATUS "${Green}Found catch2 pre-installed${ColourReset}")
    endif ()
endif ()