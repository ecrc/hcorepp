//
// Created by amr on 22/10/2022.
//

#ifndef HCOREPP_HELPERS_TYPE_CHECK_H
#define HCOREPP_HELPERS_TYPE_CHECK_H

#include <complex>

template<typename T>
struct is_complex_t : public std::false_type {
};

template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {
};

template<typename T>
constexpr bool is_complex() {
    return is_complex_t<T>::value;
}

template<typename T>
struct is_complex_float_t : public std::false_type {
};

template<>
struct is_complex_float_t<std::complex<float>> : public std::true_type {
};

template<typename T>
constexpr bool is_complex_float() {
    return is_complex_float_t<T>::value;
}

template<typename T>
struct is_double_t : public std::false_type {
};

template<>
struct is_double_t<double> : public std::true_type {
};

template<typename T>
constexpr bool is_double() {
    return is_double_t<T>::value;
}

#endif //HCOREPP_HELPERS_TYPE_CHECK_H
