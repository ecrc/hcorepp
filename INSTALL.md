HCore++ Installation Notes
--------------------------------------------------------------------------------

- [Synopsis](#synopsis)
- [Environment variables](#environment-variables)
- [QuickStart](#quickstart)
- [Requirements](#requirements)
- [Options](#options)

Synopsis
--------------------------------------------------------------------------------

Use CMake to configure and compile the HCore++ library and its tester, then
install the headers and library.

    mkdir build && cd build
    cmake ..
    make && make install

Environment variables
--------------------------------------------------------------------------------

Standard environment variables affect CMake. These include:

    CXX                C++ compiler
    CXXFLAGS           C++ compiler flags
    LDFLAGS            linker flags
    CPATH              compiler include search path
    LIBRARY_PATH       compile time library search path
    LD_LIBRARY_PATH    runtime library search path
    DYLD_LIBRARY_PATH  runtime library search path on macOS

QuickStart
--------------------------------------------------------------------------------

The CMake script enforces an out-of-source build. Create a build directory under
the HCore++ root directory:

    cd /path/to/hcorepp
    mkdir build && cd build
    cmake [-DCMAKE_INSTALL_PREFIX=/path/to/install] [options] ..
    make
    make install

If HCore++ test suite is built, then HCore++ tester can be run:

    hcorepp/build$ cd tests
    hcorepp/build/tests$ ./hcorepp-tests [-h|--help] [parameters]

Requirements
--------------------------------------------------------------------------------

HCore++ requires BLAS++ (https://bitbucket.org/icl/blaspp) and LAPACK++
(https://bitbucket.org/icl/lapackpp).
It inherits its dependencies from them, and they must be installed first via
CMake, before running HCore++'s CMake. Therefore, HCore++ should find BLAS++ and
LAPACK++, if it is installed in a system default location (e.g., `/usr/local`),
or their install prefix is the same. If HCore++ can't find them, you can
point to their directory:

    cmake -DCMAKE_PREFIX_PATH=/path/to/install [options] ..

or

    cmake -Dblaspp_DIR=/path/to/blaspp/build \
          -Dlapackpp_DIR=/path/to/lapackpp/build [options] ..

However, if CMake doesn't find BLAS++ and LAPACK++, they will be downloaded and
compiled.
See the BLAS++ [INSTALL.md](https://bitbucket.org/icl/blaspp/src/master/INSTALL.md)
and LAPACK++ [INSTALL.md](https://bitbucket.org/icl/lapackpp/src/master/INSTALL.md)
for their options. They can be specified on the command line using `-Doption=value`
syntax (not as environment variables), such as:

    cmake -Dblas=mkl -DLAPACK_LIBRARIES='-lopenblas' ..

Options
--------------------------------------------------------------------------------

HCore++ specific options include (all values are case insensitive):

    USE_CUDA
        Whether to use CUDA, if available. One of:
        yes  
        no   (default)

    HCOREPP_BUILD_TESTS
        Whether to build test suite. One of:
        yes  (default)
        no

    HCOREPP_BUILD_EXAMPLES
        Whether to build examples suite. One of:
        yes  (default)
        no

    HCOREPP_BUILD_DOCS
        Whether to build documentation. One of:
        yes  (default)
        no

    
Besides the environment variables and options listed above, additional
standard CMake options include:

    BUILD_SHARED_LIBS
        Whether to build as a static or shared library. One of:
        yes  shared library (default)
        no   static library

    CMAKE_INSTALL_PREFIX (alias prefix)
        Where to install, default /opt/slate.
        Headers go   in ${prefix}/include,
        library goes in ${prefix}/lib

    CMAKE_PREFIX_PATH
        Where to look for CMake packages such as BLAS++, LAPACK++ and
        TestSweeper.

    CMAKE_BUILD_TYPE
        Type of build. One of:
        [empty]         default compiler optimization          (no flags)
        Debug           no optimization, with asserts          (-O0 -g)
        Release         optimized, no asserts, no debug info   (-O3 -DNDEBUG)
        RelWithDebInfo  optimized, no asserts, with debug info (-O2 -DNDEBUG -g)
        MinSizeRel      Release, but optimized for size        (-Os -DNDEBUG)

    CMAKE_MESSAGE_LOG_LEVEL (alias log)
        Level of messages to report. In ascending order:
        FATAL_ERROR, SEND_ERROR, WARNING, AUTHOR_WARNING, DEPRECATION,
        NOTICE, STATUS, VERBOSE, DEBUG, TRACE.
        Particularly, DEBUG or TRACE gives useful information.

With CMake, options are specified on the command line using
`-Doption=value` syntax (not as environment variables), such as:

    # in build directory
    cmake -Dblas=openblas -DHCOREPP_BUILD_TESTS=no -DCMAKE_INSTALL_PREFIX=/usr/local ..

Alternatively, use the `ccmake` text-based interface or the CMake app GUI.

    # in build directory
    ccmake ..
    # Type 'c' to configure, then 'g' to generate Makefile

To re-configure CMake, you may need to delete CMake's cache:

    # in build directory
    rm CMakeCache.txt
    # or
    rm -rf *
    cmake [options] ..

To debug the build, set `VERBOSE`:

    # in build directory, after running cmake
    make VERBOSE=1
