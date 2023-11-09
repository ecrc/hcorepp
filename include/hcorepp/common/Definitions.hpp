/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_COMMON_DEFINITIONS_HPP
#define HCOREPP_COMMON_DEFINITIONS_HPP

// Macro definition to instantiate the HCore template classes with supported types.
#define HCOREPP_INSTANTIATE_CLASS(TEMPLATE_CLASS)   template class TEMPLATE_CLASS<float>;  \
                                                    template class TEMPLATE_CLASS<double>;
//                                                    template class TEMPLATE_CLASS<complex<double>>; \
//                                                    template class TEMPLATE_CLASS<complex<float>>;

namespace hcorepp {
    namespace common {
        /**
         * @brief
         * Enum denoting the side of some linear algebra operations, ie: unmqr
         */
        enum SideMode {
            SIDE_LEFT = 0,
            SIDE_RIGHT = 1
        };

        /**
         * @brief
         * Enum denoting the types of norms that can be requested from lapack lange functions.
         */
        enum class Norm {
            MAX = 'M',
            ONE = '1',
            INF = 'i',
            FROBENIUS = 'f'
        };

        /**
         * @brief
         * Enum denoting the blas operations that are supported to be done on the matrix.
         */
        enum BlasOperation {
            OP_NoTRANS = 0,
            OP_TRANS = 1,
            OP_C = 2,
            OP_HERMITAN = 2, /* synonym if CUBLAS_OP_C */
            OP_CONJG = 3     /* conjugate, placeholder - not supported in the current release */
        };

        /**
         * @brief
         * Enum denoting the job type for some lapack API calls following their interface.
         */
        enum class Job {
#ifdef USE_SYCL
            NoVec = 0,
            Vec = 'V',  // geev, syev, ...
            UpdateVec = 'U',  // gghrd#, hbtrd, hgeqz#, hseqr#, ... (many compq or compz)

            AllVec = 1,  // gesvd, gesdd, gejsv#
            SomeVec = 3,  // gesvd, gesdd, gejsv#, gesvj#
            OverwriteVec = 2,  // gesvd, gesdd

            CompactVec = 'P',  // bdsdc
            SomeVecTol = 'C',  // gesvj
            VecJacobi = 'J',  // gejsv
            Workspace = 'W',  // gejsv
#else
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
#endif
        };

        /**
         * @brief
         * Enum denoting the matrix type that we are operating on.
         */
        enum class MatrixType {
            General = 'G',
            Lower = 'L',
            Upper = 'U',
            Hessenberg = 'H',
            LowerBand = 'B',
            UpperBand = 'Q',
            Band = 'Z',
            };

        /**
         * @brief
         * Enum denoting some operation types that are supported as alternatives.
         */
        enum CompressionType {
            LAPACK_GESVD,
            LAPACK_GESDD
        };

        /**
         * @brief
         * The supported memory allocation strategies.
         */
        enum class MemoryHandlerStrategy {
            ONDEMAND,
            POOL
        };


    }//namespace common
}//namespace hcorepp

#endif //HCOREPP_COMMON_DEFINITIONS_HPP
