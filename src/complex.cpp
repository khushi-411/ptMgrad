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
    complex(const _X& __r): __re_(__r), __im_(0) {}

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
    void real(_X __re) const {
        __re_ = __re;
    }
    
    template <class _X> constexpr
    void imag(_X __im) const {
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
};


template<>
class complex<float> {
private:
    typedef float U;
    U __re_;
    U __im_;
public:
    constexpr
    complex(const float& __r): __re_(__r), __im_(0) {}

    constexpr
    complex(const complex<float>& __c): __re_(__c.real()), __im_(__c.imag()) {}

    constexpr float real() const {
        return __re_;
    }

    constexpr float imag() const {
        return __im_;
    }

    template <class _X> constexpr
    void real(_X __re) const {
        __re_ = __re;
    }

    template <class _X> constexpr
    void imag(_X __im) const {
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
};


template<>
class complex<double> {
private:
    typedef double U;
    U __re_;
    U __im_;
public:
    constexpr complex(const double& __r): __re_(__r), __im_(0) {}

    constexpr complex(const complex<double>& __c): __re_(__c.real()), __im_(__c.imag()) {}

    constexpr double real() const {
        return __re_;
    }

    constexpr double imag() const {
        return __im_;
    }

    template <class _X> constexpr
    void real(_X __re) const {
        __re_ = __re;
    }

    template <class _X> constexpr
    void imag(_X __im) const {
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
};


}  // namespace ptMgrad
