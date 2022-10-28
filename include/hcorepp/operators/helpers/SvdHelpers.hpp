
#ifndef HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP
#define HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP

#include <cstdint>

namespace hcorepp {
    namespace helpers {
        enum SideMode {
            SIDE_LEFT = 0,
            SIDE_RIGHT = 1
        };

        enum class Norm {
            MAX = 'M',
            ONE = '1',
            INF = 'i',
            FROBENIUS = 'f'
        };

        enum BlasOperation {
            OP_NoTRANS = 0,
            OP_TRANS = 1,
            OP_C = 2,
            OP_HERMITAN = 2, /* synonym if CUBLAS_OP_C */
            OP_CONJG = 3     /* conjugate, placeholder - not supported in the current release */
        };

        enum class Job {
            NoVec = 'N',
            Vec = 'V',  // geev, syev, ...
            UpdateVec = 'U',  // gghrd#, hbtrd, hgeqz#, hseqr#, ... (many compq or compz)

            AllVec = 'A',  // gesvd, gesdd, gejsv#
            SomeVec = 'S',  // gesvd, gesdd, gejsv#, gesvj#
            OverwriteVec = 'O',  // gesvd, gesdd

            CompactVec = 'P',  // bdsdc
            SomeVecTol = 'C',  // gesvj
            VecJacobi = 'J',  // gejsv
            Workspace = 'W',  // gejsv
        };

        enum class MatrixType {
            General = 'G',
            Lower = 'L',
            Upper = 'U',
            Hessenberg = 'H',
            LowerBand = 'B',
            UpperBand = 'Q',
            Band = 'Z',
        };

        enum operationType {
            LAPACK_GESVD,
            LAPACK_GESDD
        };

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
                       int64_t aFixedRank = 0, operationType aOpType = LAPACK_GESVD);

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

            operationType
            GetOperationType() const;

        private:
            bool mUseTrmm = false;
            bool mUseUngqr = true;
            bool mTruncatedSvd = false;
            int64_t mFixedRank = 0;
            operationType mOpType = LAPACK_GESVD;
        };
    }
}

#endif //HCOREPP_OPERATORS_HELPERS_SVD_HELPERS_HPP
