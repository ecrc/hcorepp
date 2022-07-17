
#ifndef HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP
#define HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP

#include <cstdint>

namespace hcorepp {
    namespace helpers {

        class SvdHelpers {
        public:

            SvdHelpers(bool aUseTrmm = false, bool aUseUngqr = true, bool aTruncatedSvd = false,
                       int64_t aFixedRank = 0);

            ~SvdHelpers();

            bool
            GetTrmm() const;

            bool
            GetUngqr() const;

            bool
            GetTruncatedSvd() const;

            int64_t
            GetFixedRank() const;

        private:
            bool mUseTrmm = false;
            bool mUseUngqr = true;
            bool mTruncatedSvd = false;
            int64_t mFixedRank = 0;
        };
    }
}

#endif //HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP
