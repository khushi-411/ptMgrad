// write a tiny autograd engine in C++
// implement backpropagation over a dynamically
// built DAG and a small neural network

// create a Value class to store a single scalar
// value and its gradient

#pragma once

#include <vector>
#include <iostream>
#include <set>
#include <functional>
#include <cmath>

#include "complex.h"


namespace ptMgrad {


template <typename T>
class Value {
private:
    T x;
    T y;
    mutable T grad = T(0);
    std::vector<Value<T>*> children;
    std::function<void()> backward_fn = []() {};

public:
    // default constructor
    Value() : x(T(0.0)), y(T(0.0)) {}

    template <class _X> constexpr
    Value(const Value<_X>& _x) : x(_x) {}

    template <class _X> constexpr
    Value(const Value<_X>& _x, const Value<_X>& _y) : x(_x), y(_y) {}

    // for scalar values
    template <class _X> constexpr
    Value(const _X& _x) : x(_x) {}

    template <class _X> constexpr
    Value(const _X& _x, const _X& _y) : x(_x), y(_y) {}

    void add_child(const Value<T>* child) {
        children.push_back(const_cast<Value<T>*>(child));
    }

    void add_grad(const T& _grad) const {
        grad += _grad;
    }

    template <class _X> constexpr
    Value& operator=(const Value<_X>& x) {
        this->x = x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator=(const _X& x) {
        this->x = x;
        return *this;
    }

    T dataX() const {
        return x;
    }

    T dataY() const {
        return y;
    }

    T gradX() const {
        return grad;
    }

    T gradY() const {
        return grad;
    }

    T get_grad() const {
        return grad;
    }

    // void add_grad(const T& _grad) const {
    //     grad += _grad;
    // }

    template <class _X> constexpr
    Value& operator- () {
        this->x = -x;
        this->y = -y;
        return *this;
    }

    constexpr operator bool() const {
        return x || y;
    }

    // add backward function implementation
    void backward() {
        // topological order all of the children in the graph
        std::vector<Value<T>*> topo;
        std::set<Value<T>*> visited;

        std::function<void(Value<T>*)> build_topo = [&](Value<T>* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto* child : v->children) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };
        build_topo(this);

        // go one variable at a time and apply the chain rule to get its gradient
        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward_fn();
        }
    }

    void set_backward(const std::function<void()>& fn) {
        backward_fn = fn;
    }

    template <class _X> constexpr
    Value& operator+= (const Value<_X>& x) {
        this->x += x.x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator+= (const _X& x) {
        this->x += x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator-= (const Value<_X>& x) {
        this->x -= x.x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator-= (const _X& x) {
        this->x -= x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator*= (const Value<_X>& x) {
        this->x *= x.x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator*= (const _X& x) {
        this->x *= x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator/= (const Value<_X>& x) {
        this->x /= x.x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator/= (const _X& x) {
        this->x /= x;
        return *this;
    }

    // TODO: why const?
    template <class _X> constexpr
    bool operator<(const Value<_X>& _x) const {  // Note: return type bool
        return x < _x.x;
    }

    template <class _X> constexpr
    bool operator<(const _X& _x) {
        return x < _x;
    }

    // TODO: why we don't need operator>

    friend std::ostream& operator<< (std::ostream& os, const Value& _x) {
        os << _x.x;
        return os;
    }
};


// template <typename T>
// T add_grad(const T& v, const T& _grad) {
//     T grad = 0.0;
//     grad = v + _grad;
//     return grad;
// }


template <class T>
inline
Value <T>
operator+ (const Value<T>& x, const Value<T>& y) {
    Value<T> __k = x;
    __k += y;
    
    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        // x.grad = add_grad(x.get_grad(), __k.get_grad());
        // y.grad = add_grad(y.get_grad(), __k.get_grad());
        x.add_grad(__k.get_grad());
        y.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator+ (const Value<T>& x, const T& y) {
    Value<T> __k = x;
    __k += y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        // x.grad = add_grad(x.get_grad(), __k.get_grad());
        x.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator+ (const T& x, const T& y) {
    Value<T> __k = x;
    __k += y;
    return __k;
}


template <class T>
inline
Value <T>
operator+ (const Value<complex<T>>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k += y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
        y.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator+ (const Value<complex<T>>& x, const complex<T>& y) {
    Value<complex<T>> __k = x;
    __k += y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator+ (const complex<T>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k += y;

    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        y.add_grad(__k.get_grad());
    });

    return __k;
}


template <class T>
inline
Value <T>
operator- (const Value<T>& x, const Value<T>& y) {
    Value<T> __k = x;
    __k -= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
        y.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator- (const Value<T>& x, const T& y) {
    Value<T> __k = x;
    __k -= y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator- (const T& x, const T& y) {
    Value<T> __k = x;
    __k -= y;
    return __k;
}


template <class T>
inline
Value <T>
operator- (const Value<complex<T>>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k -= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
        y.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator- (const Value<complex<T>>& x, const complex<T>& y) {
    Value<complex<T>> __k = x;
    __k -= y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T>
inline
Value <T>
operator- (const complex<T>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k -= y;

    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        y.add_grad(__k.get_grad());
    });

    return __k;
}


template <class T>
inline
Value <T>
operator* (const Value<T>& x, const Value<T>& y) {
    Value<T> __k = x;
    __k *= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() * y);
        y.add_grad(__k.get_grad() * x);
    });

    return __k;
}

template <class T>
inline
Value <T>
operator* (const Value<T>& x, const T& y) {
    Value<T> __k = x;
    __k *= y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() * y);
    });

    return __k;
}

template <class T>
inline
Value <T>
operator* (const T& x, const T& y) {
    Value<T> __k = x;
    __k *= y;
    return __k;
}


template <class T>
inline
Value <T>
operator* (const Value<complex<T>>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k *= y;

    __k.add_child(&x);
    __k.add_child(&y);

    // TODO: fix this
    //__k.set_backward([&x, &y, &__k]() {
    //    x.add_grad(__k.get_grad() * y);
    //    y.add_grad(__k.get_grad() * x);
    //});

    return __k;
}

template <class T>
inline
Value <T>
operator* (const Value<complex<T>>& x, const complex<T>& y) {
    Value<complex<T>> __k = x;
    __k *= y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() * y);
    });

    return __k;
}

template <class T>
inline
Value <T>
operator* (const complex<T>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k *= y;

    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        y.add_grad(__k.get_grad() * x);
    });

    return __k;
}


template <class T>
inline
Value <T>
operator/ (const Value<T>& x, const Value<T>& y) {
    Value<T> __k = x;
    __k /= y;
    return __k;
}

template <class T>
inline
Value <T>
operator/ (const Value<T>& x, const T& y) {
    Value<T> __k = x;
    __k /= y;
    return __k;
}

template <class T>
inline
Value <T>
operator/ (const T& x, const T& y) {
    Value<T> __k = x;
    __k /= y;
    return __k;
}


template <class T>
inline
Value <T>
operator/ (const Value<complex<T>>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k /= y;
    return __k;
}


template <class T>
inline
Value <T>
operator/ (const complex<T>& x, const Value<complex<T>>& y) {
    Value<complex<T>> __k = x;
    __k /= y;
    return __k;
}

template <class T>
inline
Value <T>
operator/ (const Value<complex<T>>& x, const complex<T>& y) {
    Value<complex<T>> __k = x;
    __k /= y;
    return __k;
}


template <class T>
inline
Value <T>
operator-(const Value<T>& _x) {
    return Value<T>(-_x);
}

template <class T>
inline
Value <T>
operator-(const Value<complex<T>>& _x) {
    return Value<complex<T>>(-_x);
}


// Either keep this or the operator< member function in Value class
/*
template <class T>
inline
bool
operator<(const Value<T>& _x, const Value<T>& _y) {
    bool __k;
    __k = _x.dataX() < _y.dataX();
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
    __k = _x.dataX() < _y;
    return __k;
}
*/

// add

template <class T>
inline
Value <T>
add(const Value<T>& _x, const Value<T>& _y) {
    return _x + _y;
}

template <class T>
inline
Value <T>
add(const Value<T>& _x, const T& _y) {
    return _x + _y;
}

template <class T>
inline
Value <T>
add(const T& _x, const T& _y) {
    return _x + _y;
}


template <class T>
inline
std::vector<Value<T>>
add(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] + _y[i]);
    }
    return __k;
}


template <class T>
inline
std::vector<Value<T>>
add(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] + _y);
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
add(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(add(_x[i], _y[i]));
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
add(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(add(_x[i], _y));
    }
    return __k;
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
Value <T>
sub(const Value<T>& _x, const Value<T>& _y) {
    return _x - _y;
}

template <class T>
inline
Value <T>
sub(const Value<T>& _x, const T& _y) {
    return _x - _y;
}

template <class T>
inline
Value <T>
sub(const T& _x, const T& _y) {
    return _x - _y;
}


template <class T>
inline
std::vector<Value<T>>
sub(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] - _y[i]);
    }
    return __k;
}


template <class T>
inline
std::vector<Value<T>>
sub(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] - _y);
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
sub(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(sub(_x[i], _y[i]));
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
sub(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(sub(_x[i], _y));
    }
    return __k;
}


// rsub

template <class T>
inline
Value <T>
rsub(const Value<T>& _x, const Value<T>& _y) {
    return _y - _x;
}

template <class T>
inline
Value <T>
rsub(const Value<T>& _x, const T& _y) {
    return _y - _x.dataX();
}

template <class T>
inline
Value <T>
rsub(const T& _x, const T& _y) {
    return _y - _x;
}


template <class T>
inline
std::vector<Value<T>>
rsub(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_y[i] - _x[i]);
    }
    return __k;
}


template <class T>
inline
std::vector<Value<T>>
rsub(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_y - _x[i]);
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
rsub(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(rsub(_x[i], _y[i]));
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
rsub(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(rsub(_x[i], _y));
    }
    return __k;
}


// mul

template <class T>
inline
Value <T>
mul(const Value<T>& _x, const Value<T>& _y) {
    return _x * _y;
}

template <class T>
inline
Value <T>
mul(const Value<T>& _x, const T& _y) {
    return _x * _y;
}

template <class T>
inline
Value <T>
mul(const T& _x, const T& _y) {
    return _x * _y;
}


template <class T>
inline
std::vector<Value<T>>
mul(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] * _y[i]);
    }
    return __k;
}


template <class T>
inline
std::vector<Value<T>>
mul(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] * _y);
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<Value<T>>>
mul(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(mul(_x[i], _y[i]));
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
mul(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(mul(_x[i], _y));
    }
    return __k;
}


// div

template <class T>
inline
Value <T>
div(const Value<T>& _x, const Value<T>& _y) {
    return _x / _y;
}

template <class T>
inline
Value <T>
div(const Value<T>& _x, const T& _y) {
    return _x / _y;
}

template <class T>
inline
Value <T>
div(const T& _x, const T& _y) {
    return _x / _y;
}


template <class T>
inline
std::vector<Value<T>>
div(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] / _y[i]);
    }
    return __k;
}


template <class T>
inline
std::vector<Value<T>>
div(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] / _y);
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<Value<T>>>
div(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(div(_x[i], _y[i]));
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
div(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(div(_x[i], _y));
    }
    return __k;
}


// rdiv

template <class T>
inline
Value <T>
rdiv(const Value<T>& _x, const Value<T>& _y) {
    return _y / _x;
}

template <class T>
inline
Value <T>
rdiv(const Value<T>& _x, const T& _y) {
    return _y / _x.dataX();
}

template <class T>
inline
Value <T>
rdiv(const T& _x, const T& _y) {
    return _y / _x;
}

template <class T>
inline
std::vector<Value<T>>
rdiv(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_y[i] / _x[i]);
    }
    return __k;
}


template <class T>
inline
std::vector<Value<T>>
rdiv(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_y / _x[i]);
    }
    return __k;
}


// pow

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


// neg

template <class T>
inline
Value <T>
neg(const Value<T>& _x) {
    return Value<T>(-_x.dataX());
}

template <class T>
inline
Value <T>
neg(const T& _x) {
    return Value<T>(-_x);
}

template <class T>
inline
std::vector<Value<T>>
neg(const std::vector<Value<T>>& _x) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(-_x[i].dataX());
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<Value<T>>>
neg(const std::vector<std::vector<Value<T>>>& _x) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(neg(_x[i]));
    }
    return __k;
}

template <class T>
inline
Value<ptMgrad::complex<T>>
neg(const Value<ptMgrad::complex<T>>& _x) {
    return Value<complex<T>>(complex<T>(-_x.dataX().real(), -_x.dataX().imag()));
}

template <class T>
inline
std::vector<Value<ptMgrad::complex<T>>>
neg(const std::vector<Value<ptMgrad::complex<T>>>& _x) {
    std::vector<Value<complex<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(
            Value<complex<T>>(complex<T>(-_x[i].dataX().real(), -_x[i].dataX().imag()))
        );
    }
    return __k;
}


// lt

template <class T>    return Value<ptMgrad::complex<T>>(-_x.dataX());

inline
bool
lt(const Value<T>& _x, const Value<T>& _y) {
    return _x < _y;
}

template <class T>
inline
bool
lt(const Value<T>& _x, const T& _y) {
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
std::vector<bool>
lt(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<bool> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] < _y[i]);
    }
    return __k;
}

template <class T>
inline
std::vector<bool>
lt(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<bool> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] < _y);
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<bool>>
lt(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<bool>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(lt(_x[i], _y[i]));
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<bool>>
lt(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<bool>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(lt(_x[i], _y));
    }
    return __k;
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
gt(const Value<T>& _x, const T& _y) {
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
std::vector<bool>
gt(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<bool> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] > _y[i]);
    }
    return __k;
}

template <class T>
inline
std::vector<bool>
gt(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<bool> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] > _y);
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<bool>>
gt(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<bool>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(gt(_x[i], _y[i]));
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<bool>>
gt(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<bool>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(gt(_x[i], _y));
    }
    return __k;
}


// relu

template <class T>
inline
Value <T>
relu(const Value<T>& _x) {
    if (_x.dataX() < 0.0) {
        return Value<T>(0.0);
    } else {
        return Value<T>(_x.dataX());
    }
}

template <class T>
inline
Value <T>
relu(const T& _x) {
    if (_x < 0.0) {
        return Value<T>(0.0);
    } else {
        return Value<T>(_x);
    }
}


template <class T>
inline
std::vector<Value<T>>
relu(const std::vector<Value<T>>& _x) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (auto& x : _x) {
        __k.push_back(relu(x));
    }
    return __k;
}


template <class T>
inline
std::vector<std::vector<Value<T>>>
relu(const std::vector<std::vector<Value<T>>>& _x) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (auto& x_row : _x) {
        __k.push_back(relu(x_row));
    }
    return __k;
}

}
