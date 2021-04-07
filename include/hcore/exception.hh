// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_EXCEPTION_HH
#define HCORE_EXCEPTION_HH

#include <string>
#include <cstdio>
#include <cstdarg>
#include <exception>

namespace hcore {

class Error : public std::exception {
public:
    Error() : std::exception()
    {
    }
    Error(const std::string& what_arg) : std::exception(), what_arg_(what_arg)
    {
    }
    Error(const std::string& what_arg, const char* func) : std::exception(),
        what_arg_(what_arg + " in function " + func)
    {
    }
    virtual const char* what() const noexcept override
    {
        return what_arg_.c_str();
    }

private:
    std::string what_arg_;
};

namespace internal {

// internal helper function; throws Error if cond is true
// called by hcore_error_if macro
inline void throw_if(bool cond, const char* condstr, const char* func)
{
    if (cond) {
        throw Error(condstr, func);
    }
}
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by hcore_error_if_msg macro
// condstr is ignored, but differentiates this from other version.
inline void throw_if(
    bool cond, const char* condstr, const char* func, const char* format, ...)
    __attribute__((format(printf, 4, 5)));

inline void throw_if(
    bool cond, const char* condstr, const char* func, const char* format, ...)
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start(va, format);
        vsnprintf(buf, sizeof(buf), format, va);
        throw Error(buf, func);
    }
}

// internal helper function; aborts if cond is true
// uses printf-style format for error message
// called by hcore_error_if_msg macro
inline void abort_if(bool cond, const char* func,  const char* format, ...)
    __attribute__((format( printf, 3, 4 )));

inline void abort_if(bool cond, const char* func,  const char* format, ...)
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start(va, format);
        vsnprintf(buf, sizeof(buf), format, va);
        fprintf(stderr, "Error: %s, in function %s\n", buf, func);
        abort();
    }
}

} // namespace internal
} // namespace hcore

#if defined(HCORE_ERROR_NDEBUG) || \
   (defined(HCORE_ERROR_ASSERT) && defined(NDEBUG))
    // HCORE does no error checking, and thus errors maybe either handled by
    //     - BLAS++ and LAPACK++, or
    //     - Lower level BLAS and LAPACK via xerbla
    #define hcore_error_if(cond)          ((void)0)
    #define hcore_error_if_msg(cond, ...) ((void)0)
#elif defined(HCORE_ERROR_ASSERT)
    // HCORE aborts on error (similar to C/C++ assert)
    #define hcore_error_if(cond) \
        hcore::internal::abort_if(cond, __func__, "%s", #cond)
    #define hcore_error_if_msg(cond, ...) \
        hcore::internal::abort_if(cond, __func__, __VA_ARGS__)
#else
    // HCORE throws errors (default)
    // internal macro to get string #cond; throws Error if cond is true
    // example: hcore_error_if(a < b)
    #define hcore_error_if(cond) \
        hcore::internal::throw_if(cond, #cond, __func__)
    // internal macro takes cond and printf-style format for error message.
    // throws Error if cond is true.
    // example: hcore_error_if_msg(a < b, "a %d < b %d", a, b);
    #define hcore_error_if_msg(cond, ...) \
        hcore::internal::throw_if(cond, #cond, __func__, __VA_ARGS__)
#endif

#endif // HCORE_EXCEPTION_HH
