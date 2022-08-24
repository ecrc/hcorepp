
#include <hcorepp/operators/helpers/SvdHelpers.hpp>

namespace hcorepp {
    namespace helpers {
        SvdHelpers::SvdHelpers(bool aUseTrmm, bool aUseUngqr, bool aTruncatedSvd, int64_t aFixedRank,
                               operationType aOpType) {
            mUseTrmm = aUseTrmm;
            mUseUngqr = aUseUngqr;
            mTruncatedSvd = aTruncatedSvd;
            mFixedRank = aFixedRank;
            mOpType = aOpType;
        }

        SvdHelpers::~SvdHelpers() {

        }

        bool SvdHelpers::GetTrmm() const {
            return mUseTrmm;
        }

        bool SvdHelpers::GetUngqr() const {
            return mUseUngqr;
        }

        bool SvdHelpers::GetTruncatedSvd() const {
            return mTruncatedSvd;
        }

        int64_t SvdHelpers::GetFixedRank() const {
            return mFixedRank;
        }

        operationType SvdHelpers::GetOperationType() const {
            return mOpType;
        }
    }
}
