// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_EXCEPTION_HH
#define HCORE_EXCEPTION_HH

#include <exception>
#include <cstdlib>
#include <string>
#include <cstdio>

namespace hcore {

//==============================================================================
//
class Error : public std::exception {
public:
    //--------------------------------------------------------------------------
    /// Constructs empty HCORE error
    Error() : std::exception() {}

    //--------------------------------------------------------------------------
    /// Constructs HCORE error with message
    ///
    /// @param[in] what_arg
    ///     Explanatory string.
    Error(const std::string& what_arg) : std::exception(), what_arg_(what_arg)
    {}

    //--------------------------------------------------------------------------
    /// Constructs HCORE error with message
    ///
    /// @param[in] what_arg
    ///     Explanatory string.
    /// @param[in] func
    ///     Function name.
    /// @param[in] file
    ///     File name.
    /// @param[in] line
    ///     Line number.
    Error(const std::string& what_arg, const char* func, const char* file,
          int line)
        : std::exception(),
          what_arg_(what_arg
                    + ", function " + func
                    + ", file "     + file
                    + ", line "     + std::to_string(line)
                    + ".")
    {}

    //--------------------------------------------------------------------------
    virtual const char* what() const noexcept override
    {
        return what_arg_.c_str();
    }

private:
    std::string what_arg_;

}; // class Error
}  // namespace hcore

#if defined(HCORE_ERR_NDEBUG) || (defined(HCORE_ERR_ASSERT) && defined(NDEBUG))
    // HCORE does no error checking, and thus errors maybe either handled by
    // BLAS++ and LAPACK++, or Lower level BLAS and LAPACK via xerbla
    #define hcore_assert(x) ((void)0)
#elif defined(HCORE_ERR_ASSERT)
    // HCORE aborts on error (similar to C/C++ assert)
    #define hcore_assert(x)                                                  \
        do {                                                                 \
            if (!(x)) {                                                      \
                fprintf(stderr, "HCORE assertion failed: (%s), "             \
                                "function %s, file %s, line %d.\n",          \
                                #x, __func__, __FILE__, __LINE__);           \
                abort();                                                     \
            }                                                                \
        } while (0)
#else
    #define hcore_assert(x)                                                  \
        do {                                                                 \
            if (!(x)) {                                                      \
                throw Error("HCORE exception thrown: ("+std::string(#x)+")", \
                            __func__, __FILE__, __LINE__);                   \
            }                                                                \
        } while (0)
#endif

#endif // HCORE_EXCEPTION_HH
