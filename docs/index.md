# What is HCore++
HCORE++ is convenient, performance-oriented C++ software API for tile low-rank matrix algebra. HCORE implements BLAS functionality in the form of tile routines; update one or a small number of individual tiles, generally sequentially on a single compute unit. Notably, an m-by-n matrix is a collection of individual mb-by-nb tiles. HCORE tiles are first C++ class objects, which are entities that can be individually allocated, destroyed, and passed to low-level tile routines, e.g., GEMM. HCORE tile routines rely on the tile low-rank compression, which replaces the dense operations with the equivalent low rank operations, to reduce the memory footprint and/or the time-to-solution.

# Features
* GEMM Operation
* Float and double precision support
* x86 and CUDA support

# Installation

Installation requires `CMake` of version 3.21.2 at least. To build HCore++, follow these instructions:

1.  Get HCore++ from git repository
```
git clone git@github.com:ecrc/hcorepp
```

2.  Go into Hcore++ folder
```
cd hcorepp
```

3.  Create build directory and go there
```
mkdir build && cd build
```

4.  Use CMake, if no installed BLAS library is found, it will install openBLAS as fallback. You can also choose whether to build x86 support or CUDA support.
```
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install/ -DUSE_CUDA=ON/OFF
```

5.  Build Hcore++
```
make -j
```

6.  Build local documentation (optional)
```
make docs
```

7.  Install Hcore++
```
make install
```
8. Add line to your .bashrc file to use Hcore++ as a library.
```
        export PKG_CONFIG_PATH=/path/to/install:$PKG_CONFIG_PATH
```
    
Now you can use `pkg-config` executable to collect compiler and linker flags for Hcore++.

# Quick Start

## Matrix Multiplication

You can find it under examples/matrix_multiplication showcasing how to do matrix multiplication for matrices in a tile format, using HCore GEMM API.

