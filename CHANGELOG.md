v1.0.0 | 2021.04.11
--------------------------------------------------------------------------------

   - Modern C++ design
      - Templates for precision to minimize codebase
         - Instantiation for 4 precisions (float, double, complex, complex-double)
      - Exceptions to avoid silently ignoring errors
      - Use standard containers: std::{vector, arrays, list, tuple, set, pair, map}
      - Use shared pointers to enforce shallow copy semantics (no deep copies)
   - Support flexible and non-uniform tile sizes
   - Rely on BLAS++ and LAPACK++ to provide a C++ API for vendor-optimized BLAS and LAPACK math libraries
   - Support all combinations of GEMM
      - ddd, ddc, dcd, dcc, cdd, cdc, ccd, and ccc, where c is compressed tile and d is dense tile
   - Provide a complete build script generator based on CMake
   - Provide a comprehensive testing suite based on TestSweeper framework

v0.1.1 | 2021.02.03
--------------------------------------------------------------------------------

   - GEMMM in the form of: C = alpha x A (compressed) x B (compressed) + beta x C
   - Decompress a low-rank tile: A = U x VT
   - Complex-double precision

v0.1.0 | 2020.01.08
--------------------------------------------------------------------------------

   - GEMMM in the form of: C (compressed) = alpha x A (compressed) x B (compressed) + beta x C (compressed)
   - SYRK in the form of: C = alpha x A (compressed) A^T (compressed) x beta x C
   - Double precision
   - Testing suite
