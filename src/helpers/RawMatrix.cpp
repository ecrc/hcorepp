/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

//#include <mkl_lapack.h>
#include <hcorepp/helpers/RawMatrix.hpp>
#include <hcorepp/common/Definitions.hpp>
//#include <lapacke_utils.h>
#include <hcorepp/helpers/LapackWrappers.hpp>
#include <cstring>
#include <cblas.h>
#include "hcorepp/kernels/RunContext.hpp"
#include "hcorepp/data-units/DataHolder.hpp"

namespace hcorepp {
    namespace helpers {

        template<typename T>
        RawMatrix<T>::RawMatrix(size_t aM, size_t aN,
                                const generators::Generator<T> &aGenerator) : mpData(nullptr), mM(0), mN(0) {
            this->mpData = (T *) malloc(aM * aN * sizeof(T));
            this->mM = aM;
            this->mN = aN;
            aGenerator.GenerateValues(aM, aN, aM, mpData);
        }

        template<typename T>
        RawMatrix<T>::RawMatrix(size_t aM, size_t aN) : mpData(nullptr), mM(0), mN(0) {
            this->mpData = (T *) malloc(aM * aN * sizeof(T));
            this->mM = aM;
            this->mN = aN;
            memset(this->mpData, 0, aM * aN * sizeof(T));
        }

        template<typename T>
        RawMatrix<T>::RawMatrix(int64_t aM, int64_t aN, T *apData) : mM(0), mN(0) {
            this->mpData = (T *) malloc(aM * aN * sizeof(T));
            memcpy(this->mpData, apData, aM * aN * sizeof(T));
            this->mM = aM;
            this->mN = aN;
        }

        template<typename T>
        blas::real_type<T> RawMatrix<T>::Norm() {
            return lapack_lange(hcorepp::common::Norm::INF, this->mM, this->mN,
                                this->mpData, this->mM);
        }

        template<typename T>
        void RawMatrix<T>::ReferenceDifference(const RawMatrix<T> &aReferenceMatrix) {
            if (this->mM != aReferenceMatrix.GetM() || this->mN != aReferenceMatrix.GetN()) {
                throw std::runtime_error("Reference difference must be called on matrices from equivalent sizes");
            }
            for (size_t i = 0; i < this->mM * this->mN; i++) {
                this->mpData[i] = aReferenceMatrix.GetData()[i] - this->mpData[i];
            }
        }

        template<typename T>
        size_t RawMatrix<T>::GetMemoryFootprint() {
            return this->mM * this->mN * sizeof(T);
        }

        template<typename T>
        RawMatrix<T> RawMatrix<T>::Clone() {
            RawMatrix<T> ret(this->mM, this->mN);
            memcpy(ret.mpData, this->mpData, this->mM * this->mN * sizeof(T));
            return ret;
        }

        template<typename T>
        void RawMatrix<T>::Print(std::ostream &aOutputStream) {
            for (size_t i = 0; i < this->mM * this->mN; i++) {
                aOutputStream << this->mpData[i] << std::endl;
            }
        }

        template<typename T>
        RawMatrix<T>::~RawMatrix() {
            if (this->mpData != nullptr) {
                free(this->mpData);
            }
        }

        template<typename T>
        T RawMatrix<T>::Normmest(T *work) {

//            printf("NormEst input array \n");
//            for (int i = 0; i < mM * mN; i++) {
//                printf("%f, ", mpData[i]);
//            }
//            printf("\n");

//            T *X = (T *) calloc(mM * 2, sizeof(T));

            T e0, tol, normx, normsx, alpha, beta;
            int i, cnt, maxiter;
            T *X = work;
            T *SX;
            SX = X + mM;

//            printf("M = %d N= %d \n", mM, mN);
            T e;
            char norm = 'I';
//            this->Norm();

//            LAPACK_dlange((const char *) &norm, (const int *) &mM, (const int *) &mN, (double *) mpData, (const int *) &mM,
//                          (double *) X,1);

            lapack_int lwork = std::max((lapack_int) 1, (lapack_int) mM);
            std::vector<blas::real_type<T>> t(lwork);

            dlange_((const char *) &norm, (const lapack_int *) &mM, (const lapack_int *) &mN, (double *) mpData,
                    (const lapack_int *) &mM,
                    (double *) X, lwork);

//            printf("NormEst A array \n");
//            for (int i = 0; i < mM * mN; i++) {
//                printf("%f, ", mpData[i]);
//            }
//            printf("\n");
//
//            printf("NormEst X array \n");
//            for (int x = 0; x < mM * 2; x++) {
//                printf("%f, ", X[x]);
//            }
//            printf("\n");
            //            lapack_lange(hcorepp::common::Norm::INF, mM, mN, mpData, mM);

//            printf(" N= %d \n", mN);

            e = cblas_dnrm2(mN, (double *) X, 1);

//            printf(" First e calculated = %f \n", e);

            if (e == 0.0) {
                return e;
            } else {
                normx = e;
            }
            alpha = 1.;
            tol = 1.e-1;
            cnt = 0;
            e0 = 0.0;
            maxiter = (100 < mN) ? 100 : mN;
            while ((cnt < maxiter) &&
                   (fabs((e) - e0) > (tol * (e)))) {
                e0 = e;

                /**
                 * Scale x = x / ||A||
                 */

                alpha = 1.0 / e;
                i = 1;
                //dscal_(&M, &alpha, X, &i);
//                printf("e[0]  = %f \n",e);

//                printf("Alpha  = %f \n",alpha);

//                printf(" X array before scal\n");
//                for (int x = 0; x < mM * 2; x++) {
//                    printf("%f, ", X[x]);
//                }
//                printf("\n");

                cblas_dscal(mM, alpha, (double *) X, i);

//                printf(" X array after scal\n");
//                for (int x = 0; x < mM * 2; x++) {
//                    printf("%f, ", X[x]);
//                }
//                printf("\n");

                /**
                 *  Compute Sx = S * x
                 */
                e0 = e;
                alpha = 1.0;
                beta = 0.0;
                i = 1;
                //dgemv_("n", &M, &N, &alpha, A, &M, X, &i, &beta, SX, &i);
                cblas_dgemv(CblasColMajor, CblasNoTrans, mM, mN, alpha, (const double *) mpData, mM, (const double *) X,
                            i, beta, (double *) SX, i);

                /**
                 *  Compute x = S' * S * x = S' * Sx
                 */
                alpha = 1.0;
                beta = 0.0;
                i = 1;
                //dgemv_("t", &M, &N, &alpha, A, &M, SX, &i, &beta, X, &i);
                cblas_dgemv(CblasColMajor, CblasTrans, mM, mN, alpha, (const double *) mpData, mM, (const double *) SX,
                            i, beta, (double *) X, i);

                /**
                 * Compute ||x||, ||Sx||
                 */
                normx = cblas_dnrm2(mM, (double *) X, 1);
                normsx = cblas_dnrm2(mM, (double *) SX, 1);

//                printf(" normx = %lf \t normsx = %lf \n", normx, normsx);

                e = normx / normsx;
//                printf(" e calculated = %f \n", e);

                cnt++;
            }

            if ((cnt >= maxiter) &&
                (fabs((e) - e0) > (tol * (e)))) {
                fprintf(stderr, "mkl_dtwonm_Tile_Async: didn't converge\n");
            }

            return e;
        }

        HCOREPP_INSTANTIATE_CLASS(RawMatrix)

    }//namespace helpers
}//namespace hcorepp

