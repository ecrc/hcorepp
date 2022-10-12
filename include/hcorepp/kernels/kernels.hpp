#ifndef HCOREPP_API_KERNELS_HPP
#define HCOREPP_API_KERNELS_HPP

//#ifdef USE_CUDA
//
//#else
//#endif

#include "blas.hh"
#include "lapack/wrappers.hh"

namespace hcorepp {
    namespace kernels {

        template<typename T>
        void Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aM, int64_t aN, int64_t aK,
                  T &aAlpha, T const *apA, int64_t aLdA, T const *apB, int64_t aLdB, T &aBeta, T *apC, int64_t aLdC);

        template<typename T>
        void MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, T &aAlpha);

        template<typename T>
        void ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV, int64_t aLdcV,
                             T *V, int64_t aArank, const T *apBdata);

        template<typename T>
        void CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma, int64_t sizeS,
                              blas::real_type<T> &accuracy);

        template<typename T>
        void CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew);

        template<typename T>
        void CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma, T *apVTnew,
                            int64_t aSizeS, int64_t aVm);

        template<typename T>
        void CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr);

        template<typename T>
        void FillIdentityMatrix(int64_t aNumOfElements, T *apMatrix);

        template<typename T>
        void LaCpy(lapack::MatrixType aType, int64_t aM, int64_t aRank, T *apCU, int64_t aLD, T *apU, int64_t aUm);

        template<typename T>
        void Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau);

        template<typename T>
        void Laset(lapack::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                   T *apA, int64_t aLdA);

        template<typename T>
        void Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag,
                  int64_t aM, int64_t aN, T aAlpha, T const *apA, int64_t aLdA, T *apB, int64_t aLdB);

        template<typename T>
        void Gesvd(lapack::Job aJobu, lapack::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apS, T *apU,
                   int64_t aLdU, T *apVT, int64_t aLdVt);

        template<typename T>
        void
        Unmqr(lapack::Side aSide, lapack::Op aTrans, int64_t aM, int64_t aN, int64_t aK, T const *apA, int64_t aLdA,
              T const *apTau, T *apC, int64_t aLdC);

        template<typename T>
        blas::real_type<T> *AllocateSigma(int64_t aSizeS);

        template<typename T>
        void DestroySigma(blas::real_type<T> *apSigma);

        template<typename T>
        T *AllocateArray(int64_t aRows, int64_t aCols, T *apSrc = nullptr);

        template<typename T>
        void DestroyArray(T *apArray);

        template<typename T>
        void Memcpy(T *apDestination, T *apSrcDataArray, int64_t aNumOfElements);


    }
}

#endif //HCOREPP_API_KERNELS_HPP
