HCORE Example
================================================================================

This is designed as a minimal, standalone example to demonstrate
how to include and link with HCORE. This assumes that HCORE has
been compiled and installed. There are two options:

## Option 1: CMake

CMake must know the compiler used to compile HCORE. Set CXX to the
compiler, in your environment.

It's best (but not required) to compile out-of-source in a build directory:

    mkdir build && cd build

If HCORE is installed outside the default search path, tell cmake
where, for example, in /opt/hicma:

    cmake -DCMAKE_PREFIX_PATH=/opt/hicma ..

Otherwise, simply run:

    cmake ..

Then, to build `example_gemm_ccc` using the resulting Makefile, run:

    make

## Option 2: Makefile

The Makefile must know the compiler used to compile HCORE,
CXXFLAGS, and LIBS. Set CXX to the compiler, either in your environment
or in the Makefile. For the flags, the CXXFLAGS and LIBS for HCORE are
hard-coded in the Makefile.

Then, to build `example_gemm_ccc` using the Makefile, run:

    make
