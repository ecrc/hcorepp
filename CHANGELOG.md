v1.0.0 | 2021.11.14
--------------------------------------------------------------------------------

   - Modern C++ design
      - Templates for precision to minimize codebase
         - Instantiation for float, double, complex, and complex-double
      - Exceptions to avoid silently ignoring errors
      - Rely upon standard containers. e.g., std::vector
   - Use BLAS++ and LAPACK++ as abstraction layers to provide C++ APIs for the
   Basic Linear Algebra Subroutines (BLAS) and LAPACK (Linear Algebra PACKage)
   - Support flexible and non-uniform tile sizes
   - Support all possible combinations of general matrix-matrix multiplication (GEMM)
```
C (dense)      = alpha x A (dense)      x B (dense)      + beta x C (dense)
C (dense)      = alpha x A (compressed) x B (dense)      + beta x C (dense)
C (dense)      = alpha x A (dense)      x B (compressed) + beta x C (dense)
C (dense)      = alpha x A (compressed) x B (compressed) + beta x C (dense)
C (compressed) = alpha x A (dense)      x B (dense)      + beta x C (compressed)
C (compressed) = alpha x A (compressed) x B (dense)      + beta x C (compressed)
C (compressed) = alpha x A (dense)      x B (compressed) + beta x C (compressed)
C (compressed) = alpha x A (compressed) x B (compressed) + beta x C (compressed)
```
   - Support row-major and column-major
   - Support no transpose, transpose, and conjugate-transpose
```c++
auto AT = transpose(A);
auto BT = conj_transpose(B);
// gemm( layout, Op::Trans, Op::ConjTrans, alpha, A, B, beta, C );
hcore::gemm<T>( alpha, AT, BT, beta, C );
```
   - Provide a complete build script generator based on CMake
   - Provide a comprehensive integration testing suite and unite testing suite
   based on TestSweeper framework
   - Support symmetric rank-k update (SYRK) and triangular matrix solve (TRSM)
   - Support Cholesky factorization (POTRF)
