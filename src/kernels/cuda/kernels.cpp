#include <cuda_runtime.h>
#include <iostream>
#include <hcorepp/kernels/kernels.hpp>
#include <cstring>
#include <cublas_v2.h>
#include "hcorepp/kernels/cuda/CudaKernels.hpp"
//#include <cusolverDn.h>
//#include "lapack/device.hh"
//#include <hcorepp/kernels/cuda/CudaKernels.hpp>

namespace hcorepp {
    namespace kernels {

        template<typename T>
        void Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aM, int64_t aN, int64_t aK,
                  T &aAlpha, T const *apA, int64_t aLdA, T const *apB, int64_t aLdB, T &aBeta, T *apC, int64_t aLdC) {

//            T *dA = blas::device_malloc<T>(aM * aK);  // m-by-k
//            T *dB = blas::device_malloc<T>(aK * aN);  // k-by-n
//            T *dC = blas::device_malloc<T>(aM * aN);  // m-by-n

//            std::cout << " M = " << aM << " N = " << aN << " aK = " << aK << "\n";
//            T *host_a = (T *) malloc(aM * aK * sizeof(T));
//            T *host_b = (T *) malloc(aK * aN * sizeof(T));
//            T *host_c = (T *) malloc(aM * aN * sizeof(T));

//            cudaMemcpy(host_a, apA, aM * aK * sizeof(T), cudaMemcpyDeviceToHost);
//            cudaMemcpy(host_b, apB, aK * aN * sizeof(T), cudaMemcpyDeviceToHost);
//            cudaMemcpy(host_c, apC, aM * aN * sizeof(T), cudaMemcpyDeviceToHost);
            int device = 0;

            int batch_size = 1000;  // todo: use default batch_size

            blas::Queue queue(device, batch_size);

//            blas::device_memcpy(dA, host_a, aM * aK, queue);
//            blas::device_memcpy(dB, host_b, aK * aN, queue);
//            blas::device_memcpy(dC, host_c, aM * aN, queue);

            blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, apA, aLdA, apB, aLdB, aBeta, apC, aLdC, queue);

//            blas::device_getmatrix(aM, aN, dC, aLdC, host_c, aLdC, queue);

            queue.sync();

//            cudaMemcpy(apC, host_c, aM * aN * sizeof(T), cudaMemcpyHostToDevice);

//            blas::device_free(dA);
//            dA = nullptr;
//            blas::device_free(dB);
//            dB = nullptr;
//            blas::device_free(dC);
//            dC = nullptr;
//            free(host_a);
//            free(host_b);
//            free(host_c);

        }

        template
        void Gemm(blas::Layout, blas::Op, blas::Op, int64_t, int64_t, int64_t,
                  float &, float const *, int64_t, float const *, int64_t, float &, float *, int64_t);

        template
        void Gemm(blas::Layout, blas::Op, blas::Op, int64_t, int64_t, int64_t,
                  double &, double const *, int64_t, double const *, int64_t, double &, double *, int64_t);

        template<typename T>
        void
        MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, T &aAlpha) {
            hcorepp::cudakernels::MultiplyByAlpha(apArray, aRows, aCols, aM, aRank, aAlpha);
        }

        template
        void
        MultiplyByAlpha(float *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, float &aAlpha);

        template
        void
        MultiplyByAlpha(double *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, double &aAlpha);

        template<typename T>
        void
        ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV, int64_t aLdcV, T *V,
                        int64_t aArank, const T *apBdata) {
            hcorepp::cudakernels::ProcessVpointer(aN, aCRank, aGetUngqr, Vm, aBeta, apCV, aLdcV, V, aArank, apBdata);
        }

        template
        void
        ProcessVpointer(int64_t, int64_t, bool, int64_t, float &, float *, int64_t, float *, int64_t, const float *);

        template
        void
        ProcessVpointer(int64_t, int64_t, bool, int64_t, double &, double *, int64_t, double *, int64_t,
                        const double *);

        template<typename T>
        void CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma, int64_t sizeS,
                              blas::real_type<T> &accuracy) {
            hcorepp::cudakernels::CalculateNewRank<T>(aNewRank, aTruncatedSvd, apSigma, sizeS, accuracy);
        }

        template
        void CalculateNewRank<float>(int64_t &, bool, blas::real_type<float> *, int64_t, blas::real_type<float> &);

        template
        void CalculateNewRank<double>(int64_t &, bool, blas::real_type<double> *, int64_t, blas::real_type<double> &);

        template<typename T>
        void CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew) {
            hcorepp::cudakernels::CalculateUVptr(aRank, aVm, UVptr, Vnew);
        }

        template
        void CalculateUVptr(int64_t, int64_t, float *, const float *);

        template
        void CalculateUVptr(int64_t, int64_t, double *, const double *);

        template<typename T>
        void
        CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma, T *apVTnew,
                       int64_t aSizeS, int64_t aVm) {
            hcorepp::cudakernels::CalculateVTnew(aRkNew, aUngqr, aMinVmVn, apSigma, apVTnew, aSizeS, aVm);
        }

        template
        void
        CalculateVTnew(int64_t, bool, int64_t, blas::real_type<float> *, float *, int64_t, int64_t);

        template
        void
        CalculateVTnew(int64_t, bool, int64_t, blas::real_type<double> *, double *, int64_t, int64_t);

        template<typename T>
        void
        CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr) {
            hcorepp::cudakernels::CalculateUVptrConj(aRank, aVm, UVptr);
        }

        template
        void CalculateUVptrConj(int64_t, int64_t, float *);

        template
        void CalculateUVptrConj(int64_t, int64_t, double *);

        template<typename T>
        void
        FillIdentityMatrix(int64_t aNumOfElements, T *apMatrix) {
            hcorepp::cudakernels::GenerateIdentityMatrix(aNumOfElements, apMatrix);
        }

        template
        void
        FillIdentityMatrix(int64_t aNumOfElements, float *apMatrix);

        template
        void
        FillIdentityMatrix(int64_t aNumOfElements, double *apMatrix);

        template<typename T>
        void
        LaCpy(helpers::MatrixType aType, int64_t aM, int64_t aRank, T *apCU, int64_t aLD, T *apU, int64_t aUm) {
            hcorepp::cudakernels::LaCpy(aType, aM, aRank, apCU, aLD, apU, aUm);
        }

        template
        void
        LaCpy(helpers::MatrixType, int64_t, int64_t, float *, int64_t, float *, int64_t);

        template
        void
        LaCpy(helpers::MatrixType, int64_t, int64_t, double *, int64_t, double *, int64_t);

        template<typename T>
        void Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau) {
            hcorepp::cudakernels::Geqrf(aM, aN, apA, aLdA, apTau);
        }

        template
        void Geqrf(int64_t, int64_t, float *, int64_t, float *);

        template
        void Geqrf(int64_t, int64_t, double *, int64_t, double *);

        template<typename T>
        void Laset(helpers::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                   T *apA, int64_t aLdA) {
            hcorepp::cudakernels::Laset(aMatrixType, aM, aN, aOffdiag, aDiag, apA, aLdA);
        }

        template
        void Laset(helpers::MatrixType aMatrixType, int64_t aM, int64_t aN, float aOffdiag, float aDiag,
                   float *apA, int64_t aLdA);

        template
        void Laset(helpers::MatrixType aMatrixType, int64_t aM, int64_t aN, double aOffdiag, double aDiag,
                   double *apA, int64_t aLdA);

        template<typename T>
        void Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag,
                  int64_t aM, int64_t aN, T aAlpha, T const *apA, int64_t aLdA, T *apB, int64_t aLdB) {
            blas::trmm(aLayout, aSide, aUplo, aTrans, aDiag, aM, aN, aAlpha, apA, aLdA, apB, aLdB);
        }

        template
        void Trmm(blas::Layout, blas::Side, blas::Uplo, blas::Op, blas::Diag, int64_t, int64_t, float,
                  float const *, int64_t, float *, int64_t);

        template
        void Trmm(blas::Layout, blas::Side, blas::Uplo, blas::Op, blas::Diag, int64_t, int64_t, double,
                  double const *, int64_t, double *, int64_t);

        template<typename T>
        void
        Gesvd(helpers::Job aJobu, helpers::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apS, T *apU,
              int64_t aLdU, T *apVT, int64_t aLdVt) {
            hcorepp::cudakernels::Gesvd(aJobu, aJobvt, aM, aN, apA, aLdA, apS, apU, aLdU, apVT, aLdVt);
        }

        template
        void
        Gesvd(helpers::Job, helpers::Job, int64_t, int64_t, float *, int64_t, float *, float *, int64_t, float *,
              int64_t);

        template
        void
        Gesvd(helpers::Job, helpers::Job, int64_t, int64_t, double *, int64_t, double *, double *, int64_t, double *,
              int64_t);

        template<typename T>
        void
        Unmqr(helpers::SideMode aSide, helpers::BlasOperation aTrans, int64_t aM, int64_t aN, int64_t aK, T const *apA,
              int64_t aLdA, T const *apTau, T *apC, int64_t aLdC) {
            hcorepp::cudakernels::Unmqr(aSide, aTrans, aM, aN, aK, apA, aLdA, apTau, apC, aLdC);
        }

        template
        void
        Unmqr(helpers::SideMode, helpers::BlasOperation, int64_t, int64_t, int64_t, float const *, int64_t,
              float const *, float *, int64_t);

        template
        void
        Unmqr(helpers::SideMode, helpers::BlasOperation, int64_t, int64_t, int64_t, double const *, int64_t,
              double const *, double *, int64_t);

        template<typename T>
        blas::real_type<T> *AllocateSigma(int64_t aSizeS) {
            blas::real_type<T> *sigma;
            cudaMalloc((void **) &sigma, aSizeS * sizeof(blas::real_type<T>));

            return sigma;
        }

        template
        blas::real_type<float> *AllocateSigma<float>(int64_t);

        template
        blas::real_type<double> *AllocateSigma<double>(int64_t);

        template<typename T>
        void DestroySigma(blas::real_type<T> *apSigma) {
            if (apSigma != nullptr) {
                cudaFree(apSigma);
            }
        }

        template
        void DestroySigma<float>(blas::real_type<float> *);

        template
        void DestroySigma<double>(blas::real_type<double> *);


        template<typename T>
        void
        ungqr(int64_t aM, int64_t aN, int64_t aK, T *apA, int64_t aLdA, T *apTau) {
            hcorepp::cudakernels::ungqr(aM, aN, aK, apA, aLdA, apTau);
        }

        template
        void
        ungqr(int64_t, int64_t, int64_t, float *, int64_t, float *);

        template
        void
        ungqr(int64_t, int64_t, int64_t, double *, int64_t, double *);


    }
}
