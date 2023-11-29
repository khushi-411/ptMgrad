#pragma once
#include <iostream>

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

}
