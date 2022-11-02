/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <hcorepp/operators/helpers/SVDParameters.hpp>

namespace hcorepp {
    namespace operators {
        SVDParameters::SVDParameters(double aAccuracy, bool aUseTrmm, bool aUseUngqr,
                                     bool aTruncatedSvd, int64_t aFixedRank,
                                     common::OperationType aOpType) {
            mUseTrmm = aUseTrmm;
            mUseUngqr = aUseUngqr;
            mTruncatedSvd = aTruncatedSvd;
            mFixedRank = aFixedRank;
            mOpType = aOpType;
            mAccuracy = aAccuracy;
        }

        SVDParameters::~SVDParameters() {

        }

        bool SVDParameters::GetTrmm() const {
            return mUseTrmm;
        }

        bool SVDParameters::GetUngqr() const {
            return mUseUngqr;
        }

        bool SVDParameters::GetTruncatedSvd() const {
            return mTruncatedSvd;
        }

        int64_t SVDParameters::GetFixedRank() const {
            return mFixedRank;
        }

        common::OperationType SVDParameters::GetOperationType() const {
            return mOpType;
        }

        double SVDParameters::GetAccuracy() const {
            return this->mAccuracy;
        }
    }
}
