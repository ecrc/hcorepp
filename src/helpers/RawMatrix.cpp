/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <hcorepp/helpers/RawMatrix.hpp>
#include <hcorepp/common/Definitions.hpp>
#include <hcorepp/helpers/LapackWrappers.hpp>
#include <cstring>

namespace hcorepp {
    namespace helpers {

        template<typename T>
        RawMatrix<T>::RawMatrix(int64_t aM, int64_t aN,
                                const generators::Generator<T> &aGenerator) : mpData(nullptr), mM(0), mN(0) {
            this->mpData = (T *) malloc(aM * aN * sizeof(T));
            this->mM = aM;
            this->mN = aN;
            aGenerator.GenerateValues(aM, aN, aM, mpData);
        }

        template<typename T>
        RawMatrix<T>::RawMatrix(int64_t aM, int64_t aN) : mpData(nullptr), mM(0), mN(0) {
            this->mpData = (T *) malloc(aM * aN * sizeof(T));
            this->mM = aM;
            this->mN = aN;
            memset(this->mpData, 0, aM * aN * sizeof(T));
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
            for (int64_t i = 0; i < this->mM * this->mN; i++) {
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
        RawMatrix<T>::~RawMatrix() {
            if (this->mpData != nullptr) {
                free(this->mpData);
            }
        }

        HCOREPP_INSTANTIATE_CLASS(RawMatrix)

    }//namespace helpers
}//namespace hcorepp

