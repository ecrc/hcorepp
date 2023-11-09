/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_OPERATORS_HELPERS_COMPRESSION_PARAMETERS_HPP
#define HCOREPP_OPERATORS_HELPERS_COMPRESSION_PARAMETERS_HPP

#include <blas/util.hh>
#include <hcorepp/common/Definitions.hpp>

namespace hcorepp {
    namespace operators {
        /**
         * @brief
         * Class responsible of encapsulating all optional parameters responsible for the potential SVD
         * operation.
         */
        class CompressionParameters {
        public:

            /**
             * @brief
             * Constructor of SVD Helpers class, to be used while calling GEMM functionality for compressed tile.
             *
             * @param[in] aAccuracy
             * The SVD operation accuracy, ie: allowed numerical threshold.
             *
             * @param[in] aUseTrmm
             * Use trmm
             *
             * @param[in] aUseUngqr
             * Use ungqr with gemm
             *
             * @param[in] aTruncatedSvd
             * Truncation to fixed accuracy * tolerance.
             *
             * @param[in] aFixedRank
             * Truncation to fixed rank. fixed_rk >= 0.
             *
             * @param[in] aOpType
             * CompressionType
             */
            CompressionParameters(double aAccuracy = 1e-4, bool aUseTrmm = false, bool aUseUngqr = true,
                                  bool aTruncatedSvd = false, size_t aFixedRank = 0,
                                  common::CompressionType aOpType = common::CompressionType::LAPACK_GESDD);

            /**
             * @brief
             * SVD helpers destructor.
             */
            ~CompressionParameters();

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
            size_t
            GetFixedRank() const;

            /**
             * @brief
             * Get the type of operation that is utilized for the SVD.
             *
             * @return
             * The operation type to use if set, otherwise LAPACK_GESVD is returned by default
             */
            common::CompressionType
            GetOperationType() const;

            /**
             * @brief
             * Get the allowed numerical threshold accuracy for the SVD reduction
             * operation.
             *
             * @return
             * The configured numerical threshold accuracy.
             */
            double
            GetAccuracy() const;

        private:
            /// Flag indicating usage for TRMM operation in SVD
            bool mUseTrmm;
            /// Flag indicating usage for UNGQR operation in SVD
            bool mUseUngqr;
            /// Flag indicating usage for truncated SVD operations
            bool mTruncatedSvd;
            /// The fixed rank that is targeted by the SVD operation
            size_t mFixedRank;
            /// The type of operation used for the SVD
            common::CompressionType mOpType;
            /// Numerical error thershold
            double mAccuracy;
        };
    }//namespace operators
}//namespace hcorepp

#endif //HCOREPP_OPERATORS_HELPERS_COMPRESSION_PARAMETERS_HPP
