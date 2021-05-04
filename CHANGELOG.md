v1.0.0 | 2021.05.01
--------------------------------------------------------------------------------

   - Modern C++ design
      - Templates for precision to minimize codebase
         - Instantiation for float, double, complex, and complex-double
      - Exceptions to avoid silently ignoring errors
      - Use standard containers. e.g., std::vector
   - Use BLAS++ and LAPACK++ as abstraction layers to provide C++ APIs for the
   Basic Linear Algebra Subroutines (BLAS) and LAPACK (Linear Algebra PACKage).
   - Support flexible and non-uniform tile sizes
   - Support all combinations of GEMM
      - ddd, ddc, dcd, dcc, cdd, cdc, ccd, and ccc; where c is compressed tile
      and d is dense tile
   - Provide a complete build script generator based on CMake
   - Provide a comprehensive testing suite based on TestSweeper framework

v0.1.1 | 2021.02.03
--------------------------------------------------------------------------------

   - GEMM in the form of: C = alpha x A (compressed) x B (compressed) + beta x C
   - Decompress a low-rank tile: A = U x VT
   - Complex-double precision
   - Support dense SYRK, POTRF, and TRSM

v0.1.0 | 2020.01.08
--------------------------------------------------------------------------------

   - GEMM in the form of:
   C (compressed) = alpha x A (compressed) x B (compressed) + beta x C (compressed)
   - SYRK in the form of: C = alpha x A (compressed) A^T (compressed) x beta x C
   - Double precision
   - Testing suite
