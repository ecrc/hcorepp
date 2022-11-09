The HCore++ Library
===========================================================
HCore++ is convenient, performance-oriented C++ software API for tile low-rank matrix algebra. HCore++ implements BLAS functionality in the form of tile routines; update one or a small number of individual tiles, generally sequentially on a single compute unit. Notably, an m-by-n matrix is a collection of individual mb-by-nb tiles. HCore++ tiles are first C++ class objects, which are entities that can be individually allocated, destroyed, and passed to low-level tile routines, e.g., GEMM. HCore++ tile routines rely on the tile low-rank compression, which replaces the dense operations with the equivalent low rank operations, to reduce the memory footprint and/or the time-to-solution.


Features of HCore++ 2.0.0
-----------------------------
* Matrix Compression
* Matrix-Matrix Multiplication (Gemm)
* Single and double precision
* CUDA support
* Testing Suite

Project Hierarchy
--------------------

* **```prerequisites```**\
  Folder containing the prerequisites needed for the project, or default scripts to install them.

* **```include```**\
  Folder containing all the headers of the system, reflecting the src directory structure.

* **```src```**\
  Folder containing all the source files of the system.
  * api : Contains the high level drivers for the HCore++ functionalities that are provided to library users
  * operators : Contains the high level data structures for the HCore++ library that the library users should create to interact with the library facilities, those are separated to the tiles concrete implementation, dense and compressed, as well as the compression parameters object
  * data-units : Contains the internally used data units that are used as the basic data unit containing a matrix of elements in contiguous memory.
  * kernels : Contains the backend implementations for all the kernels and memory operations used inside the library, any changes to technology support should be added there.
  * helpers : Contains helper facilities that are useful for examples, and testing. This includes quick wrappers to create multi-tile matrices, timers, and general wrappers for lapack functionalities.

* **```tests```**\
  Folder containing all the tests of the system. Follows same structure as the src folder.

* **```examples```**\
  Folder containing all the demo code showcasing how the framework is used within applications, and the capabilities of the framework.

* **```clean_build.sh```**\
  Script used to build the system tests after running the config.sh, and by default build all the different modules of the project.

* **```config.sh```**\
  Script used to generate the building system inside a 'bin' directory.

* **```CMakeLists.txt```**\
  The top level CMake file to configure the build system.


Current Research
----------------
* Performance optimization
* Support for more BLAS operations
* Support for more hardware accelerators
* Support for complex precisions
* Auto-tuning: Tile Size, Fixed Accuracy and Fixed Ranks


External Dependencies
---------------------
HCore++ depends on the following libraries:
* BLAS
* LAPACK
* BLAS++
* LAPACK++
* CUDA toolkit(if building with CUDA support)

Installation
------------

Please see INSTALL.md for information about installing and testing.


References
-----------
1. K. Akbudak, H. Ltaief, A. Mikhalev, and D. E. Keyes, *Tile Low Rank Cholesky Factorization for
   Climate/Weather Modeling Applications on Manycore Architectures*, **International Supercomputing
   Conference (ISC17)**, June 18-22, 2017, Frankfurt, Germany.

2. K. Akbudak, H. Ltaief, A. Mikhalev, A. Charara, and D. E. Keyes, *Exploiting Data Sparsity for Large-Scale Matrix Computations*, **Euro-Par 2018**, August 27-31, 2018, Turin, Italy.

3. Q. Cao, Y. Pei, T. Herault, K. Akbudak, A. Mikhalev, G. Bosilca, H. Ltaief, D. E. Keyes, and J. Dongarra, *Performance Analysis of Tile Low-Rank Cholesky Factorization Using PaRSEC Instrumentation Tools*, **IEEE/ACM International Workshop on Programming and Performance Visualization Tools (ProTools)**, Denver, CO, USA, 2019, pp. 25-32.

4. Q. Cao, Y. Pei, K. Akbudak, A. Mikhalev, G. Bosilca, H. Ltaief, D. E. Keyes, and J. Dongarra, *Extreme-Scale Task-Based Cholesky Factorization Toward Climate and Weather Prediction Applications*, **The Platform for Advanced Scientific Computing (PASC 2020)**.

5. N. Al-Harthi, R. Alomairy, K. Akbudak, R. Chen, H. Ltaief, H. Bagci, and D. E. Keyes, *Solving Acoustic Boundary Integral Equations Using High Performance Tile Low-Rank LU Factorization*, **International Supercomputing Conference (ISC 2020)**.

6. Q. Cao, Y. Pei, K. Akbudak, G. Bosilca, H. Ltaief, D. E. Keyes, and J. Dongarra, *Leveraging PaRSEC Runtime Support to Tackle Challenging 3D Data-Sparse Matrix Problems*, **IEEE International Parallel & Distributed Processing Symposium (IPDPS 2021)**.
