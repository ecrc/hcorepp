macro(BuildDependency raw_name url tag)
    string(TOLOWER ${raw_name} name)
    string(TOUPPER ${raw_name} capital_name)
    message(STATUS "Fetching ${name} ${tag} from ${url}")
    include(FetchContent)
    FetchContent_Declare(${name} GIT_REPOSITORY "${url}" GIT_TAG "${tag}")
    FetchContent_Populate(${name})
    set(${name}_srcpath ${CMAKE_BINARY_DIR}/_deps/${name}-src)
    set(${name}_binpath ${CMAKE_BINARY_DIR}/_deps/${name}-bin)
    set(${name}_installpath ${CMAKE_BINARY_DIR}/_deps/${name}-install)
    file(MAKE_DIRECTORY ${${name}_binpath})
    file(MAKE_DIRECTORY ${${name}_installpath})
    # Configure subproject into <subproject-build-dir>
    execute_process(COMMAND ${CMAKE_COMMAND}
            -DCMAKE_INSTALL_PREFIX=${${name}_installpath}
#            -DBUILD_SHARED_LIBS=ON
            ${${name}_srcpath}
            WORKING_DIRECTORY
            ${${name}_binpath})
    # Build and install subproject
    include(ProcessorCount)
    ProcessorCount(N)
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${${name}_binpath} --parallel ${N} --target install)
    set(ENV{LD_LIBRARY_PATH} "$ENV{LD_LIBRARY_PATH}:${${name}_installpath}/lib")
    set(ENV{LIBRARY_PATH} "$ENV{LIBRARY_PATH}:${${name}_installpath}/lib")
    set(ENV{CPATH} "$ENV{CPATH}:${${name}_installpath}/include")
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${${name}_installpath}/lib/pkgconfig")
    set(${capital_name}_DIR "${${name}_installpath}/lib")
    include_directories(${${name}_installpath}/include)
    link_directories(${${name}_installpath}/lib)
endmacro()