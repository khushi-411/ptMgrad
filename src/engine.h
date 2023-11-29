#pragma once
#include <iostream>

#include "complex.h"


namespace ptMgrad {

// TODO: do we need different overloads like class Value<float>, class Value<double> etc?
template <typename T>
class Value {
private:
    typedef T U;
    U x;
    U y;
public:
    Value() {};

    // TODO: cannot use const directly in func param
    template <class _X> constexpr
    Value(const Value<_X>& _x): x(_x) {};

    template <class _X> constexpr
    Value(const Value<_X>& _x, const Value<_X>& _y): x(_x), y(_y) {}

    // for scalar values
    template <class _X> constexpr
    Value(const _X& _x): x(_x) {}

    template <class _X> constexpr
    Value(const _X& _x, const _X& _y): x(_x), y(_y) {}

    template <class _X> constexpr
    Value& operator= (const Value<_X>& _x) {
        // PS: "this" is not mandatory, i guess
        this->x = _x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator= (const _X& _x) {
        this->x = _x;
        return *this;
    }

    T dataX() const {
        return x;
    }

    T dataY() const {
        return y;
    }

    template <class _X> constexpr
    Value& operator- () {
        this->x = -x;
        this->y = -y;
        return *this;
    }

    constexpr operator bool() const {
        return x || y;
    }

    template <class _X> constexpr
    Value& operator+= (const Value <_X>& _x) {
        this->x += _x.x;  // TODO: remove ".x"
        return *this;
    }

    template <class _X> constexpr
    Value& operator+= (const _X& _x) {
        this->x += _x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator-=(const Value<_X>& _x) {
        this->x -= _x.x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator-=(const _X& _x) {
        this->x -= _x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator*=(const Value<_X>& _x) {
        this->x *= _x.x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator*=(const _X& _x) {
        this->x *= _x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator/=(const Value<_X>& _x) {
        this->x /= _x.x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator/=(const _X& _x) {
        this->x /= _x;
        return *this;
    }

    // TODO: why const?
    // TODO: Figure out why we don't need this
    /*template <class _X> constexpr
    bool operator<(const Value<_X>& _x) const {  // Note: return type bool
        return x < _x.x;
    }

    template <class _X> constexpr
    bool operator<(const _X& _x) {
        return x < _x;
    }*/

    friend std::ostream& operator<<(std::ostream& os, const Value& _x) {
        os << _x.x;
        return os;
    }
};

// We'll need these if we remove bool operator from Value class
// it might return bool if we remove this
template <class T>
inline
Value <T>
operator+(const Value<T>& _x, const Value<T>& _y) {
    Value<T> __k = _x;
    __k += _y;
    return __k;
}

template <class T>
inline
Value <T>
operator+(const T& _x, const T& _y) {
    Value<T> __k = _x;
    __k += _y;
    return __k;
}

template <class T>
inline
Value <T>
operator+(const Value<T>& _x, const T&  _y) {
    Value<T> __k = _x;
    __k += _y;
    return __k;
}

/*
template <class T>
inline
Value<ptMgrad::complex<T>>
operator+(const Value<ptMgrad::complex<T>>& _x, const Value<ptMgrad::complex<T>>& _y) {
    Value<ptMgrad::complex<T>> __k =  _x;
    __k += _y;
    return __k;
}

template <class T>
inline
Value<ptMgrad::complex<T>>
operator+(const Value<ptMgrad::complex<T>>& _x, const T& _y) {
    Value<ptMgrad::complex<T>> __k =  _x;
    __k += _y;
    return __k;
}


template <class T>
inline
Value<ptMgrad::complex<T>>
operator+(const T& _x, const Value<ptMgrad::complex<T>>& _y) {
    Value<ptMgrad::complex<T>> __k =  _y;
    __k += _x;
    return __k;
}
*/

template <class T>
inline
Value <T>
operator-(const Value<T>& _x, const Value<T>& _y) {
    Value<T> __k = _x;
    __k -= _y;
    return __k;
}

template <class T>
inline
Value <T>
operator-(const T& _x, const T& _y) {
    Value<T> __k = _x;
    __k -= _y;
    return __k;
}

template <class T>
inline
Value <T>
operator-(const Value<T>& _x, const T& _y) {
    Value<T> __k = _x;
    __k -= _y;
    return __k;
}


template <class T>
inline
Value <T>
operator*(const Value<T>& _x, const Value<T>& _y) {
    Value<T> __k = _x;
    __k *= _y;
    return __k;
}

template <class T>
inline
Value <T>
operator*(const T& _x, const T& _y) {
    Value<T> __k = _x;
    __k *= _y;
    return __k;
}

template <class T>
inline
Value <T>
operator*(const Value<T>& _x, const T& _y) {
    Value<T> __k = _x;
    __k *= _y;
    return __k;
}


template <class T>
inline
Value <T>
operator/(const Value<T>& _x, const Value<T>& _y) {
    Value<T> __k = _x;
    __k /= _y;
    return __k;
}

template <class T>
inline
Value <T>
operator/(const T& _x, const T& _y) {
    Value<T> __k = _x;
    __k /= _y;
    return __k;
}

template <class T>
inline
Value <T>
operator/(const Value<T>& _x, const T& _y) {
    Value<T> __k = _x;
    __k /= _y;
    return __k;
}


template <class T>
inline
Value <T>
operator-(const Value<T>& _x) {
    return Value<T>(-_x);
}


// TODO: figure out the exact reason why we don't need this
// most probably in this case we don't need this because it already returns bool
// other case don't return bool therefore we need those
// if we add these operator overloading and use "lt(a, b)" it causes the following:
// Program returned: 139: Program terminated with signal: SIGSEGV
/*
template <class T>
inline
bool
operator<(const Value<T>& _x, const Value<T>& _y) {
    bool __k;
    __k = _x < _y;
    return __k;
}

template <class T>
inline
bool
operator<(const T& _x, const T& _y) {
    bool __k;
    __k = _x < _y;
    return __k;
}

template <class T>
inline
bool
operator<(const Value<T>& _x, const T& _y) {
    bool __k;
    __k = _x < _y;
    return __k;
}
*/


// add

// TODO: maybe use class inheritance here
template <class T>
inline
Value<T>
add(const Value<T>&_x, const Value<T>& _y) {
    return _x + _y;
}

template <class T>
inline
Value<T>
add(const Value<T>& _x, const T& _y) {
    return _x + _y;
}

template <class T>
inline
T
add(const T&_x, const T& _y) {
    return _x + _y;
}

/*
template <class T>
inline
Value<ptMgrad::complex<T>>
add(const Value<ptMgrad::complex<T>>& _x, const Value<ptMgrad::complex<T>>& _y) {
    return _x + _y;
}

template <class T>
inline
Value<ptMgrad::complex<T>>
add(const Value<ptMgrad::complex<T>>& _x, const T& _y) {
    return _x + _y;
}


template <class T>
inline
Value<complex<T>>
add(const T& _x, const Value<ptMgrad::complex<T>>& _y) {
    return _x + _y;
}
*/

// sub

template <class T>
inline
Value<T>
sub(const Value<T>&_x, const Value<T>& _y) {
    return _x - _y;
}

template <class T>
inline
Value<T>
sub(const Value<T>& _x, const T& _y) {
    return _x - _y;
}

template <class T>
inline
Value<T>
sub(const T&_x, const T& _y) {
    return _x - _y;
}


// rsub

template <class T>
inline
Value<T>
rsub(const Value<T>&_x, const Value<T>& _y) {
    return _y - _x;
}

template <class T>
inline
Value<T>
rsub(const Value<T>& _x, const T& _y) {
    return _y - _x;
}

template <class T>
inline
Value<T>
rsub(const T&_x, const T& _y) {
    return _y - _x;
}


// mul

template <class T>
inline
Value<T>
mul(const Value<T>&_x, const Value<T>& _y) {
    return _x * _y;
}

template <class T>
inline
Value<T>
mul(const Value<T>& _x, const T& _y) {
    return _x * _y;
}

template <class T>
inline
Value<T>
mul(const T&_x, const T& _y) {
    return _x * _y;
}


// div

template <class T>
inline
Value<T>
div(const Value<T>&_x, const Value<T>& _y) {
    return _x / _y;
}

template <class T>
inline
Value<T>
div(const Value<T>& _x, const T& _y) {
    return _x / _y;
}

template <class T>
inline
Value<T>
div(const T&_x, const T& _y) {
    return _x / _y;
}


// rdiv

template <class T>
inline
Value<T>
rdiv(const Value<T>&_x, const Value<T>& _y) {
    return _y / _x;
}

template <class T>
inline
Value<T>
rdiv(const Value<T>& _x, const T& _y) {
    return _y / _x;
}

template <class T>
inline
Value<T>
rdiv(const T&_x, const T& _y) {
    return _y / _x;
}


// pow
/*
template <class T>
inline
Value<T>
pow(const Value<T>& _x, const Value<T>& _y) {
    Value<T> __k;
    __k = std::pow(_x.dataX(), _y.dataX());
    return Value<T>(__k);
}


template <class T>
inline
Value <T>
pow(const Value<T>& _x, const T& _y) {
    Value<T> __k;
    __k = std::pow(_x.dataX(), _y);
    return Value<T>(__k);
}

template <class T>
inline
Value <T>
pow(const T& _x, const T& _y) {
    Value<T> __k;
    __k = std::pow(_x, _y);
    return Value<T>(__k);
}
*/

// neg

// TODO: add negation operator (neg & -)
template <class T>
inline
Value <T>
neg(const Value<T>& _x) {
    return -_x;
}

template <class T>
inline
Value <T>  // TODO: use T instead?
neg(const T& _x) {
    return -_x;
}


// lt

template <class T>
inline
bool
lt(const Value<T>& _x, const Value<T>& _y) {
    return _x < _y;
}

template <class T>
inline
bool
lt(const T& _x, const T& _y) {
    return _x < _y;
}

template <class T>
inline
bool
lt(const Value<T>& _x, const T& _y) {
    return _x < _y;
}


// gt

template <class T>
inline
bool
gt(const Value<T>& _x, const Value<T>& _y) {
    return _x > _y;
}

template <class T>
inline
bool
gt(const T& _x, const T& _y) {
    return _x > _y;
}

template <class T>
inline
bool
gt(const Value<T>& _x, const T& _y) {
    return _x > _y;
}

}
