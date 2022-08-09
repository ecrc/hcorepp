
#ifndef HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP
#define HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP

#include <cstdint>

namespace hcorepp {
    namespace helpers {

        class SvdHelpers {
        public:

            /**
             * @brief
             * Constructor of SVD Helpers class, to be used while calling GEMM functionality for compressed tile.
             *
             * @param aUseTrmm
             * Use trmm
             * @param aUseUngqr
             * Use ungqr with gemm
             * @param aTruncatedSvd
             * Truncation to fixed accuracy * tolerance.
             * @param aFixedRank
             * Truncation to fixed rank. fixed_rk >= 0.
             */
            SvdHelpers(bool aUseTrmm = false, bool aUseUngqr = true, bool aTruncatedSvd = false,
                       int64_t aFixedRank = 0);

            /**
             * @brief
             * SVD helpers destructor.
             */
            ~SvdHelpers();

            /**
             * @brief
             * Get TRMM flag.
             *
             * @return
             * UseTrmm flag.
             */
            bool
            GetTrmm() const;

            /**
             * @brief
             * Get Ungqr flag.
             *
             * @return
             * ungqr flag.
             */
            bool
            GetUngqr() const;

            /**
             * @brief
             * Get truncated SVD flag.
             *
             * @return
             * TruncatedSVD flag.
             */
            bool
            GetTruncatedSvd() const;

            /**
             * @brief
             * Get the fixed rank set.
             *
             * @return
             * Fixed rank if set, otherwise 0 is returned.
             */
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
