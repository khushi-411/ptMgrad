#pragma once

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
    complex(const _X& __re = U(), const _X& __im = U()): __re_(__re), __im_(__im) {}

    template <class _X> constexpr
    complex(const complex<_X>& __c): __re_(__c.real()), __im_(__c.imag()) {}

    template <class _X> constexpr
    _X real() const {
        return __re_;
    }

    template <class _X> constexpr
    _X imag() const {
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

    friend std::ostream& operator<<(std::ostream& os, const ptMgrad::complex<T>&  __x) {
        os << "(" << __x.real() << ", " << __x.imag() << ")";
        return os;
    }

};


template<>
class complex<float> {
private:
    typedef float U;
    U __re_;
    U __im_;
public:

    constexpr
    complex(const float& __re = U(), const float& __im = U()): __re_(__re), __im_(__im) {}

    constexpr
    complex(const complex<float>& __c): __re_(__c.real()), __im_(__c.imag()) {}

    constexpr float real() const {
        return __re_;
    }

    constexpr float imag() const {
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

    template <typename _X>
    complex<float>& operator= (const complex<_X>& __x) {
        __re_ = __x.real();
        __im_ = __x.imag();
        return *this;
    }

    template <typename _X>
    complex<float>& operator+= (const complex<_X>& __x) {
        __re_ += __x.real();
        __im_ += __x.imag();
        return *this;
    }

    template <typename _X>
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

    friend std::ostream& operator<<(std::ostream& os, const ptMgrad::complex<float>&  __x) {
        os << "(" << __x.real() << ", " << __x.imag() << ")";
        return os;
    }

};


template<>
class complex<double> {
private:
    typedef double U;
    U __re_;
    U __im_;
public:
    constexpr
    complex(const double& __re = U(), const double& __im = U()): __re_(__re), __im_(__im) {}

    constexpr complex(const complex<double>& __c): __re_(__c.real()), __im_(__c.imag()) {}

    constexpr double real() const {
        return __re_;
    }

    constexpr double imag() const {
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

    template <class _X> constexpr
    complex<double>& operator*= (const complex<_X>& __x) {
        *this = *this * complex(__x.real(), __x.imag());
        return *this;
    }

    template <class _X> constexpr
    complex<double>& operator/= (const complex<_X>& __x) {
        *this = *this / complex(__x.real(), __x.imag());
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const ptMgrad::complex<double>&  __x) {
        os << "(" << __x.real() << ", " << __x.imag() << ")";
        return os;
    }

};

template <class T>
inline
ptMgrad::complex<T>
operator+(const ptMgrad::complex<T>& _x, const ptMgrad::complex<T>& _y) {
    ptMgrad::complex<T> __k =  _x;
    __k += _y;
    return __k;
}

template <class T>
inline
ptMgrad::complex<T>
operator+(const ptMgrad::complex<T>& _x, const T& _y) {
    ptMgrad::complex<T> __k =  _x;
    __k += _y;
    return __k;
}


template <class T>
inline
ptMgrad::complex<T>
operator+(const T& _x, const ptMgrad::complex<T>& _y) {
    ptMgrad::complex<T> __k =  _y;
    __k += _x;
    return __k;
}

template <class T>
inline
ptMgrad::complex<T>
operator-(const ptMgrad::complex<T>& _x, const ptMgrad::complex<T>& _y) {
    ptMgrad::complex<T> __k =  _x;
    __k -= _y;
    return __k;
}

template <class T>
inline
ptMgrad::complex<T>
operator-(const ptMgrad::complex<T>& _x, const T& _y) {
    ptMgrad::complex<T> __k =  _x;
    __k -= _y;
    return __k;
}


template <class T>
inline
ptMgrad::complex<T>
operator-(const T& _x, const ptMgrad::complex<T>& _y) {
    ptMgrad::complex<T> __k =  _y;
    __k -= _x;
    return __k;
}

template <class T>
inline
ptMgrad::complex<T>
add(const ptMgrad::complex<T>& _x, const ptMgrad::complex<T>& _y) {
    return _x + _y;
}

template <class T>
inline
ptMgrad::complex<T>
add(const ptMgrad::complex<T>& _x, const T& _y) {
    return _x + _y;
}


template <class T>
inline
ptMgrad::complex<T>
add(const T& _x, const ptMgrad::complex<T>& _y) {
    return _x + _y;
}

template <class T>
inline
ptMgrad::complex<T>
sub(const ptMgrad::complex<T>& _x, const ptMgrad::complex<T>& _y) {
    return _x - _y;
}

template <class T>
inline
ptMgrad::complex<T>
sub(const ptMgrad::complex<T>& _x, const T& _y) {
    return _x - _y;
}


template <class T>
inline
ptMgrad::complex<T>
sub(const T& _x, const ptMgrad::complex<T>& _y) {
    return _x - _y;
}

}  // namespace ptMgrad
