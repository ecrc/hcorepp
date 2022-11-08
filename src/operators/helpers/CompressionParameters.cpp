/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/operators/helpers/CompressionParameters.hpp>

namespace hcorepp {
    namespace operators {
        CompressionParameters::CompressionParameters(double aAccuracy, bool aUseTrmm, bool aUseUngqr,
                                     bool aTruncatedSvd, int64_t aFixedRank,
                                     common::CompressionType aOpType) {
            mUseTrmm = aUseTrmm;
            mUseUngqr = aUseUngqr;
            mTruncatedSvd = aTruncatedSvd;
            mFixedRank = aFixedRank;
            mOpType = aOpType;
            mAccuracy = aAccuracy;
        }

        CompressionParameters::~CompressionParameters() {

        }

        bool CompressionParameters::GetTrmm() const {
            return mUseTrmm;
        }

        bool CompressionParameters::GetUngqr() const {
            return mUseUngqr;
        }

        bool CompressionParameters::GetTruncatedSvd() const {
            return mTruncatedSvd;
        }

        int64_t CompressionParameters::GetFixedRank() const {
            return mFixedRank;
        }

        common::CompressionType CompressionParameters::GetOperationType() const {
            return mOpType;
        }

        double CompressionParameters::GetAccuracy() const {
            return this->mAccuracy;
        }
    }
}
