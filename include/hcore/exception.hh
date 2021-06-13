// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
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
        {}
    Error(const std::string& what_arg) : std::exception(), what_arg_(what_arg)
        {}
    Error(const std::string& what_arg, const char* function) : std::exception(),
        what_arg_(what_arg + ", function " + function + ".")
        {}
    Error(const std::string& what_arg, const char* function, const char* file,
        int line) : std::exception(),
            what_arg_(what_arg + ", function " + function + ", file " + file
                               + ", line " + std::to_string(line) + ".")
        {}
    virtual const char* what() const noexcept override
        { return what_arg_.c_str(); }

private:
    std::string what_arg_;
};

namespace internal {

// throws hcore::Error if condition is true called by hcore_error_if macro
inline void throw_if(
    bool condition, const char* condition_string,
    const char* function, const char* file, int line)
{
    if (condition) {
        throw Error(condition_string, function, file, line);
    }
}
// throws hcore::Error if condition is true called by hcore_error_if_msg macro
// and uses printf-style format for error message
// condition_string is ignored, but differentiates this from other version.
inline void throw_if(
    bool condition, const char* condition_string,
    const char* function, const char* file, int line, const char* format, ...)
    __attribute__((format(printf, 6, 7)));

inline void throw_if(
    bool condition, const char* condition_string,
    const char* function, const char* file, int line, const char* format, ...)
{
    if (condition) {
        char bufffer[80];
        va_list v;
        va_start(v, format);
        vsnprintf(bufffer, sizeof(bufffer), format, v);
        va_end(v);
        throw Error(bufffer, function, file, line);
    }
}

// internal helper function; aborts if condition is true
// uses printf-style format for error message
// called by hcore_error_if_msg macro
inline void abort_if(
    bool condition,
    const char* function, const char* file, int line, const char* format, ...)
    __attribute__((format(printf, 5, 6)));

inline void abort_if(
    bool condition,
    const char* function, const char* file, int line, const char* format, ...)
{
    if (condition) {
        char bufffer[80];
        va_list v;
        va_start(v, format);
        vsnprintf(bufffer, sizeof(bufffer), format, v);
        va_end(v);
        fprintf(stderr,
            "HCORE assertion failed: (%s), function %s, file %s, line %d.\n",
            bufffer, function, file, line);
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
    #define hcore_error_if(condition)          ((void)0)
    #define hcore_error_if_msg(condition, ...) ((void)0)
#elif defined(HCORE_ERROR_ASSERT)
    // HCORE aborts on error (similar to C/C++ assert)
    #define hcore_error_if(condition) \
        hcore::internal::abort_if( \
            condition, __func__, __FILE__, __LINE__, "%s", #condition)
    #define hcore_error_if_msg(condition, ...) \
        hcore::internal::abort_if( \
            condition, __func__, __FILE__, __LINE__, __VA_ARGS__)
#else
    // HCORE throws errors (default)
    // internal macro to get string #condition; throws hcore::Error if condition
    // is true. Example: hcore_error_if(a < b)
    #define hcore_error_if(condition) \
        hcore::internal::throw_if( \
            condition, #condition, __func__, __FILE__, __LINE__)
    // internal macro takes condition and printf-style format for error message.
    // throws Error if condition is true.
    // example: hcore_error_if_msg(a < b, "a %d < b %d", a, b);
    #define hcore_error_if_msg(condition, ...) \
        hcore::internal::throw_if( \
            condition, #condition, __func__, __FILE__, __LINE__, __VA_ARGS__)
#endif

#endif // HCORE_EXCEPTION_HH
