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
    mutable T grad = 0.0;
    std::vector<Value<T>*> children;
    std::function<void()> backward_fn = []() {};

public:
    // default constructor
    Value() : x(0), y(0) {}

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

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() * y);
        y.add_grad(__k.get_grad() * x);
    });

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
lt(const Value<T>& _x, const T& _y) {
    return _x < _y;
}

template <class T>
inline
bool
lt(const T& _x, const T& _y) {
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
gt(const Value<T>& _x, const T& _y) {
    return _x > _y;
}

template <class T>
inline
bool
gt(const T& _x, const T& _y) {
    return _x > _y;
}

}
