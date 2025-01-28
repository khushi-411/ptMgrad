#pragma once

#include <iostream>
#include <cmath>

#include "engine.h"

namespace ptMgrad {

template <typename T>
class complex {
private:
    typedef T U;
    U __re_;
    U __im_;

public:
    template <class _X> constexpr
    complex(const _X& __re = U(), const _X& __im = U()) : __re_(__re), __im_(__im) {}

    template <class _X> constexpr
    complex(const complex<_X>& __c) : __re_(__c.real()), __im_(__c.imag()) {}

    // template <class _X> constexpr
    T constexpr real() const {
        return __re_;
    }

    // template <class _X> constexpr
    T constexpr imag() const {
        return __im_;
    }

    template <class _X> constexpr
    void real(_X __re) {
        __re_ = __re;
    }

    template <class _X> constexpr
    void imag(_X __im) {
        __im_ = __im;
    }

    constexpr operator bool() const {
        return real() || imag();
    }

    template <class _X> constexpr
    complex& operator= (const _X& __re) {
        __re_ = __re;
        return *this;
    }

    template <class _X> constexpr
    complex& operator+= (const _X& __re) {
        __re_ += __re;
        return *this;
    }

    template <class _X> constexpr
    complex& operator-= (const _X& __re) {
        __re_ -= __re;
        return *this;
    }

    template <class _X> constexpr
    complex& operator*= (const _X& __re) {
        __re_ *= __re;
        __im_ *= __re;
        return *this;
    }

    template <class _X> constexpr
    complex& operator/= (const _X& __re) {
        __re_ /= __re;
        __im_ /= __re;
        return *this;
    }

    template <class _X> constexpr
    complex& operator= (const complex<_X>& __x) {
        __re_ = __x.real();
        __im_ = __x.imag();
        return *this;
    }

    template <class _X> constexpr
    complex& operator+= (const complex<_X>& __x) {
        __re_ += __x.real();
        __im_ += __x.imag();
        return *this;
    }

    template <class _X> constexpr
    complex& operator-= (const complex<_X>& __x) {
        __re_ -= __x.real();
        __im_ -= __x.imag();
        return *this;
    }

    template <class _X> constexpr
    complex& operator*= (const complex<_X>& __x) {
        *this = *this * complex(__x.real(), __x.imag());
        return *this;
    }

    template <class _X> constexpr
    complex& operator/= (const complex<_X>& __x) {
        *this = *this / complex(__x.real(), __x.imag());
        return *this;
    }

    friend std::ostream& operator<< (std::ostream& os, const complex& __x) {
        os << "(" << __x.real() << ", " << __x.imag() << ")";
        return os;
    }
};


template<>
class complex<float> {
private:
    typedef float U;
    float __re_;
    float __im_;

public:
    constexpr
    complex(const float& __re = U(), const float& __im = U()) : __re_(__re), __im_(__im) {}

    constexpr
    complex(const complex<float>& __c) : __re_(__c.real()), __im_(__c.imag()) {}

    constexpr
    float real() const {
        return __re_;
    }

    constexpr
    float imag() const {
        return __im_;
    }

    template <class _X> constexpr
    void real(_X __re) {
        __re_ = __re;
    }

    template <class _X> constexpr
    void imag(_X __im) {
        __im_ = __im;
    }

    constexpr operator bool() const {
        return real() || imag();
    }

    complex<float>& operator= (const float& __re) {
        __re_ = __re;
        return *this;
    }

    complex<float>& operator+= (const float& __re) {
        __re_ += __re;
        return *this;
    }

    complex<float>& operator-= (const float& __re) {
        __re_ -= __re;
        return *this;
    }

    complex<float>& operator*= (const float& __re) {
        __re_ *= __re;
        __im_ *= __re;
        return *this;
    }

    complex<float>& operator/= (const float& __re) {
        __re_ /= __re;
        __im_ /= __re;
        return *this;
    }

    template <class _X> constexpr
    complex<float>& operator= (const complex<_X>& __x) {
        __re_ = __x.real();
        __im_ = __x.imag();
        return *this;
    }

    template <class _X> constexpr
    complex<float>& operator+= (const complex<_X>& __x) {
        __re_ += __x.real();
        __im_ += __x.imag();
        return *this;
    }

    template <class _X> constexpr
    complex<float>& operator-= (const complex<_X>& __x) {
        __re_ -= __x.real();
        __im_ -= __x.imag();
        return *this;
    }

    template <class _X> constexpr
    complex<float>& operator*= (const complex<_X>& __x) {
        *this = *this * complex(__x.real(), __x.imag());
        return *this;
    }

    template <class _X> constexpr
    complex<float>& operator/= (const complex<_X>& __x) {
        *this = *this / complex(__x.real(), __x.imag());
        return *this;
    }

    friend std::ostream& operator<< (std::ostream& os, const complex<float>& __x) {
        os << "(" << __x.real() << ", " << __x.imag() << ")";
        return os;
    }
};


template<>
class complex<double> {
private:
    typedef double U;
    double __re_;
    double __im_;

public:
    constexpr
    complex(const double& __re = U(), const double& __im = U()) : __re_(__re), __im_(__im) {}

    constexpr
    complex(const complex<double>& __c) : __re_(__c.real()), __im_(__c.imag()) {}

    constexpr
    double real() const {
        return __re_;
    }

    constexpr
    double imag() const {
        return __im_;
    }

    template <class _X> constexpr
    void real(_X __re) {
        __re_ = __re;
    }

    template <class _X> constexpr
    void imag(_X __im) {
        __im_ = __im;
    }

    constexpr operator bool() const {
        return real() || imag();
    }

    complex<double>& operator= (const double& __re) {
        __re_ = __re;
        return *this;
    }

    complex<double>& operator+= (const double& __re) {
        __re_ += __re;
        return *this;
    }

    complex<double>& operator-= (const double& __re) {
        __re_ -= __re;
        return *this;
    }

    complex<double>& operator*= (const double& __re) {
        __re_ *= __re;
        __im_ *= __re;
        return *this;
    }

    complex<double>& operator/= (const double& __re) {
        __re_ /= __re;
        __im_ /= __re;
        return *this;
    }

    template <typename _X>
    complex<double>& operator= (const complex<_X>& __x) {
        __re_ = __x.real();
        __im_ = __x.imag();
        return *this;
    }

    template <typename _X>
    complex<double>& operator+= (const complex<_X>& __x) {
        __re_ += __x.real();
        __im_ += __x.imag();
        return *this;
    }

    template <typename _X>
    complex<double>& operator-= (const complex<_X>& __x) {
        __re_ -= __x.real();
        __im_ -= __x.imag();
        return *this;
    }

    template <typename _X> constexpr
    complex<double>& operator*= (const complex<_X>& __x) {
        *this = *this * complex(__x.real(), __x.imag());
        return *this;
    }

    template <typename _X> constexpr
    complex<double>& operator/= (const complex<_X>& __x) {
        *this = *this / complex(__x.real(), __x.imag());
        return *this;
    }

    friend std::ostream& operator<< (std::ostream& os, const complex<double>& __x) {
        os << "(" << __x.real() << ", " << __x.imag() << ")";
        return os;
    }
};


template <class T>
inline
complex<T>
operator+(const complex<T>& _x, const complex<T>& _y) {
    complex<T> __k = _x;
    __k += _y;
    return __k;
}

template <class T>
inline
complex<T>
operator+(const complex<T>& _x, const T& _y) {
    complex<T> __k = _x;
    __k += _y;
    return __k;
}

template <class T>
inline
complex<T>
operator+(const T& _x, const complex<T>& _y) {
    complex<T> __k = _x;
    __k += _y;
    return __k;
}


template <class T>
inline
complex<T>
operator-(const complex<T>& _x, const complex<T>& _y) {
    complex<T> __k = _x;
    __k -= _y;
    return __k;
}

template <class T>
inline
complex<T>
operator-(const complex<T>& _x, const T& _y) {
    complex<T> __k = _x;
    __k -= _y;
    return __k;
}

template <class T>
inline
complex<T>
operator-(const T& _x, const complex<T>& _y) {
    complex<T> __k = _x;
    __k -= _y;
    return __k;
}

template <class T>
inline
complex<T>
add(const complex<T>& _x, const complex<T>& _y) {
    return _x + _y;
}

template <class T>
inline
complex<T>
add(const complex<T>& _x, const T& _y) {
    return _x + _y;
}

template <class T>
inline
complex<T>
add(const T& _x, const complex<T>& _y) {
    return _x + _y;
}


template <class T>
inline
complex<T>
sub(const complex<T>& _x, const complex<T>& _y) {
    return _x - _y;
}

template <class T>
inline
complex<T>
sub(const complex<T>& _x, const T& _y) {
    return _x - _y;
}

template <class T>
inline
complex<T>
sub(const T& _x, const complex<T>& _y) {
    return _x - _y;
}


template <class T>
inline
complex<T>
rsub(const complex<T>& _x, const complex<T>& _y) {
    return _y - _x;
}

template <class T>
inline
complex<T>
rsub(const complex<T>& _x, const T& _y) {
    return _y - _x;
}

template <class T>
inline
complex<T>
rsub(const T& _x, const complex<T>& _y) {
    return _y - _x;
}

template <class T>
inline
complex<T>
operator*(const complex<T>& _x, const complex<T>& _y) {
    T _real = _x.real() * _y.real() - _x.imag() * _y.imag();
    T _imag = (_x.real() + _x.imag()) * (_y.real() + _y.imag()) - (_x.real() * _y.real() + _x.imag() * _y.imag());
    return complex<T>(_real, _imag);
}

template <class T>
inline
complex<T>
operator*(const complex<T>& _x, const T& _y) {
    T _real = _x.real() * _y;
    T _imag = _x.imag() * _y;
    return complex<T>(_real, _imag);
}

template <class T>
inline
complex<T>
operator*(const T& _x, const complex<T>& _y) {
    T _real = _x * _y.real();
    T _imag = _x * _y.imag();
    return complex<T>(_real, _imag);
}


template <class T>
inline
complex<T>
operator/(const complex<T>& _x, const complex<T>& _y) {
    T __k1 = _x.real() * _y.real() + _x.imag() * _y.imag();
    T __k2 = _x.imag() * _y.real() - _x.real() * _y.imag();
    T __k3 = _y.real() * _y.real() + _y.imag() * _y.imag();

    if (__k3 == 0) {
        throw std::invalid_argument("Division by zero");
    }

    return complex<T>(__k1 / __k3, __k2 / __k3);
}

template <class T>
inline
complex<T>
operator/(const complex<T>& _x, const T& _y) {
    if (_y == 0) {
        throw std::invalid_argument("Division by zero");
    }

    return complex<T>(_x.real() / _y, _x.imag() / _y);
}

template <class T>
inline
complex<T>
operator/(const T& _x, const complex<T>& _y) {
    T __k1 = _x * _y.real();
    T __k2 = -_x * _y.imag();
    T __k3 = _y.real() * _y.real() + _y.imag() * _y.imag();

    if (__k3 == 0) {
        throw std::invalid_argument("Division by zero");
    }

    return complex<T>(__k1 / __k3, __k2 / __k3);
}


template <class T>
inline
complex<T>
mul(const complex<T>& _x, const complex<T>& _y) {
    return _x * _y;
}

template <class T>
inline
complex<T>
mul(const complex<T>& _x, const T& _y) {
    return _x * _y;
}

template <class T>
inline
complex<T>
mul(const T& _x, const complex<T>& _y) {
    return _x * _y;
}


template <class T>
inline
complex<T>
div(const complex<T>& _x, const complex<T>& _y) {
    return _x / _y;
}

template <class T>
inline
complex<T>
div(const complex<T>& _x, const T& _y) {
    return _x / _y;
}

template <class T>
inline
complex<T>
div(const T& _x, const complex<T>& _y) {
    return _x / _y;
}


template <class T>
inline
complex<T>
rdiv(const complex<T>& _x, const complex<T>& _y) {
    return _y / _x;
}

template <class T>
inline
complex<T>
rdiv(const complex<T>& _x, const T& _y) {
    return _y / _x;
}

template <class T>
inline
complex<T>
rdiv(const T& _x, const complex<T>& _y) {
    return _y / _x;
}

template <class T>
inline
complex<T>
pow(const complex<T>& _x, const complex<T>& _y) {
    return complex<T>(std::pow(_x.real(), _y.real()), std::pow(_x.imag(), _y.imag()));
}

template <class T>
inline
complex<T>
pow(const complex<T>& _x, const T& _y) {
    return complex<T>(std::pow(_x.real(), _y), std::pow(_x.imag(), _y));
}

template <class T>
inline
complex<T>
pow(const T& _x, const complex<T>& _y) {
    return complex<T>(std::pow(_x, _y.real()), std::pow(_x, _y.imag()));
}


template <class T>
inline
complex<T>
neg(const complex<T>& _x) {
    return complex<T>(-_x.real(), -_x.imag());
}

}
