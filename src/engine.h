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
#include <type_traits>

#include "complex.h"


namespace ptMgrad {


template <typename U>
struct is_complex : std::false_type {};

template <typename U>
struct is_complex<complex<U>> : std::true_type {};

template <typename U>
inline constexpr bool is_complex_v = is_complex<U>::value;


template <typename T>
class Value {
public:
    typedef T value_type;

private:
    T x;
    T y;
    mutable T grad = T(0);
    std::vector<Value<T>*> children;
    std::function<void()> backward_fn = []() {};

public:
    // default constructor
    Value() : x(T(0.0)), y(T(0.0)), grad(T(0.0)) {}

    template <class _X> constexpr
    Value(const Value<_X>& _x) : x(_x.dataX()), y(_x.dataY()) {}

    template <class _X> constexpr
    Value(const Value<_X>& _x, const Value<_X>& _y) : x(_x.dataX()), y(_y.dataY()) {}

    // for scalar values
    template <class _X> constexpr
    Value(const _X& _x) : x(_x), y(T(0.0)) {}

    template <class _X> constexpr
    Value(const _X& _x, const _X& _y) : x(_x), y(_y) {}

    // for complex
    template <class _X> constexpr
    Value(const complex<_X>& _x) {
        // : x(_x.real()), y(_x.imag()) {}
        if  constexpr (is_complex_v<_X>) {
            x = _x;
        } else {
            x = complex<_X>(_x.real(), _x.imag());
        }
    }

    template <class _X> constexpr
    Value(const complex<_X>& _x, const complex<_X>& _y) {
         //: x(complex<_X>(_x.real(), _y.real())), y(complex<_X>(_x.imag(), _y.imag())) {}
        if constexpr (is_complex_v<_X>) {
            x = _x;
            y = _y;
        } else {
            x = complex<_X>(_x.real(), _y.real());
            y = complex<_X>(_x.imag(), _y.imag());
        }
    }

    // void add_child(const Value<T>* child) {
    //     children.push_back(const_cast<Value<T>*>(child));
    // }

    template <typename _X>
    void add_child(const Value<_X>* child) {
        Value<T>* child_cast = new Value<T>(child->dataX(), child->dataY());
        children.push_back(child_cast);
    }

    void add_grad(const T& _grad) const {
        grad += _grad;
    }

	template <typename _X, typename U=T>
    typename std::enable_if<is_complex_v<U>, void>::type
    add_grad(_X real, _X imag) const {
        grad += T(real, imag);
    }

	void zero_grad() {
        grad = T(0.0);
    }

    void set_grad(const T& _grad) {
        grad = _grad;
    }

    template <class _X> constexpr
    Value& operator=(const Value<_X>& x) {
        // error: ‘double ptMgrad::Value<double>::x’ is private within this context
        this->x = x.dataX();
        this->y = x.dataY();
        this->grad = T(0);
        this->children.clear();
        this->backward_fn = []() {};
        return *this;
    }

    template <class _X> constexpr
    Value& operator=(const _X& x) {
        this->x = x;
        this->y = T(0);
        this->grad = T(0);
        this->children.clear();
        this->backward_fn = []() {};
        return *this;
    }

    T dataX() const {
        if constexpr (is_complex_v<T>) {
            return x;
        } else {
            return x;
        }
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
    Value operator- () const {
        return Value(-x, -y);
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
        this->grad = T(1.0);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward_fn();
        }
    }

    void set_backward(const std::function<void()>& fn) {
        backward_fn = fn;
    }

    // don't keep this
    // causing ambiguity with the global operator+ and operator-
    /*
    template <class _X> constexpr
    Value operator+ (const Value<_X>& x) const {
        this->x += x.dataX();
        this->y += x.dataY();
        return *this;
    }

    template <class _X> constexpr
    Value operator+ (const _X& x) const {
        this->x += x;
        return *this;
    }

    template <class _X> constexpr
    Value operator- (const Value<_X>& x) const {
        this->x -= x.dataX();
        this->y -= x.dataY();
        return *this;
    }

    template <class _X> constexpr
    Value operator- (const _X& x) const {
        this->x -= x;
        return *this;
    }
    */

    template <class _X> constexpr
    Value& operator+= (const Value<_X>& x) {
        this->x += x.dataX();
        this->y += x.dataY();
        return *this;
    }

    template <class _X> constexpr
    Value& operator+= (const _X& x) {
        this->x += x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator-= (const Value<_X>& x) {
        this->x -= x.dataX();
        this->y -= x.dataY();
        return *this;
    }

    template <class _X> constexpr
    Value& operator-= (const _X& x) {
        this->x -= x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator*= (const Value<_X>& x) {
        this->x *= x.dataX();
        this->y *= x.dataY();
        return *this;
    }

    template <class _X> constexpr
    Value& operator*= (const _X& x) {
        this->x *= x;
        return *this;
    }

    template <class _X> constexpr
    Value& operator/= (const Value<_X>& x) {
        this->x /= x.dataX();
        this->y /= x.dataY();
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
        return x < _x.dataX();
    }

    template <class _X> constexpr
    bool operator<(const _X& _x) {
        return x < _x;
    }

    // TODO: why we don't need operator>

    friend std::ostream& operator<< (std::ostream& os, const Value& _x) {
        os << _x.dataX();
        return os;
    }
};


// template <typename T>
// T add_grad(const T& v, const T& _grad) {
//     T grad = 0.0;
//     grad = v + _grad;
//     return grad;
// }


template <class T, class U>
using ResultType = std::common_type_t<T, U>;


template <class T>
inline
Value <T>
operator+ (const Value<T>& x, const Value<T>& y) {
    if constexpr (is_complex_v<T>) {
        T val(
            x.dataX().real() + y.dataX().real(),
            x.dataX().imag() + y.dataX().imag()
        );
		Value<T> __k(val);

		__k.add_child(&x);
		__k.add_child(&y);

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
			y.add_grad(__k.get_grad().real(), __k.get_grad().imag());
		});

        return __k;
    }

	Value<T> __k(x.dataX() + y.dataX(), x.dataY() + y.dataY());

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

template <class T, class U>
inline
Value <ResultType<T, U>>
operator+ (const Value<T>& x, const Value<U>& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.dataX().real() + y.dataX().real(),
            x.dataX().imag() + y.dataX().imag()
        );
		Value<ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		// __k.add_child(static_cast<Value<ResultType<T, U>>*>(&x));
		// __k.add_child(static_cast<Value<ResultType<T, U>>*>(&y));

        __k.add_child(&x);
        __k.add_child(&y);

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
			y.add_grad(__k.get_grad().real(), __k.get_grad().imag());
		});

        return __k;
    }

    Value<ResultType<T, U>> __k = x;
    __k += y;

    // __k.add_child(static_cast<Value<ResultType<T, U>>*>(&x));
    // __k.add_child(static_cast<Value<ResultType<T, U>>*>(&y));

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
operator+ (const Value<T>& x, const T& y) {
	if constexpr (is_complex_v<T>) {
        T val(
            x.dataX().real() + y.real(),
            x.dataX().imag() + y.imag()
        );
        Value <T> __k = Value<T>(val);

        __k.add_child(&x);
		// __k.add_child(&y);    // y is scalar

        __k.set_backward([&x, &y, &__k]() {
            x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
        });

        return __k;
    }

	Value<T> __k = x;
    __k += y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
    });

    return __k;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator+ (const Value<T>& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.dataX().real() + y.real(),
            x.dataX().imag() + y.imag()
        );
        Value<ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		__k.add_child(static_cast<Value<ResultType<T, U>>*>(&x));

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
		});

        return __k;
    }
    
	Value<ResultType<T, U>> __k = x;
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
operator+ (const T& x, const T& y) {
    if constexpr (is_complex_v<T>) {
        T val(
            x.real() + y.real(),
            x.imag() + y.imag()
        );
        return Value<T>(val);
    }

	Value<T> __k = x;
    __k += y;
    return x + y;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator+ (const T& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.real() + y.real(),
            x.imag() + y.imag()
        );
        return Value<ResultType<T, U>>(val);
    }

	Value<ResultType<T, U>> __k = x;
    __k += y;
    return __k;
}


template <class T>
inline
Value <T>
operator- (const Value<T>& x, const Value<T>& y) {
    if constexpr (is_complex_v<T>) {
        T val(
            x.dataX().real() - y.dataX().real(),
            x.dataX().imag() - y.dataX().imag()
        );
        Value <T> __k = Value<T>(val);

        __k.add_child(&x);
        __k.add_child(&y);

        __k.set_backward([&x, &y, &__k]() {
            x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
            y.add_grad(-__k.get_grad().real(), -__k.get_grad().imag());
        });

        return __k;
    }

    Value<T> __k = x;
    __k -= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
        y.add_grad(-__k.get_grad());
    });

    return __k;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator- (const Value<T>& x, const Value<U>& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.dataX().real() - y.dataX().real(),
            x.dataX().imag() - y.dataX().imag()
        );
		Value<ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		__k.add_child(&x);
		__k.add_child(&y);

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
			y.add_grad(-__k.get_grad().real(), -__k.get_grad().imag());
		});

        return __k;
    }

    Value<ResultType<T, U>> __k = x;
    __k -= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
        y.add_grad(-__k.get_grad());
    });

    return __k;
}


template <class T>
inline
Value <T>
operator- (const Value<T>& x, const T& y) {
    if constexpr (is_complex_v<T>) {
        T val(
            x.dataX().real() - y.real(),
            x.dataX().imag() - y.imag()
        );
        Value<T> __k = Value<T>(val);

		__k.add_child(&x);

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
		});

        return __k;
    }

    Value<T> __k = x;
    __k -= y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad());
    });

    return __k;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
operator- (const Value<T>& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.dataX().real() - y.real(),
            x.dataX().imag() - y.imag()
        );
        Value <ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		__k.add_child(static_cast<Value<ResultType<T, U>>*>(&x));

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(__k.get_grad().real(), __k.get_grad().imag());
		});

        return __k;
    }

    Value<ResultType<T, U>> __k = x;
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
    if constexpr (is_complex_v<T>) {
        T val(
            x.real() - y.real(),
            x.imag() - y.imag()
        );
        return Value<T>(val);
    }

    Value<T> __k = x;
    __k -= y;
    return __k;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
operator- (const T& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.real() - y.real(),
            x.imag() - y.imag()
        );
        return Value<ResultType<T, U>>(val);
    }

    Value<ResultType<T, U>> __k = x;
    __k -= y;
    return __k;
}


template <class T>
inline
Value <T>
operator* (const Value<T>& x, const Value<T>& y) {
    if constexpr (is_complex_v<T>) {
        T val(
            x.dataX().real() * y.dataX().real() - x.dataX().imag() * y.dataX().imag(),
            x.dataX().real() * y.dataX().imag() + x.dataX().imag() * y.dataX().real()
        );
        Value <T> __k = Value<T>(val);

		__k.add_child(&x);
		__k.add_child(&y);

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(
				__k.get_grad().real() * y.dataX().real() - __k.get_grad().imag() * y.dataX().imag(),
				__k.get_grad().real() * y.dataX().imag() + __k.get_grad().imag() * y.dataX().real()
			);
			y.add_grad(
				__k.get_grad().real() * x.dataX().real() - __k.get_grad().imag() * x.dataX().imag(),
				__k.get_grad().real() * x.dataX().imag() + __k.get_grad().imag() * x.dataX().real()
			);
		});

		return __k;
    }

    Value<T> __k = x;
    __k *= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() * y.dataX());
        y.add_grad(__k.get_grad() * x.dataX());
    });

    return __k;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator* (const Value<T>& x, const Value<U>& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.dataX().real() * y.dataX().real() - x.dataX().imag() * y.dataX().imag(),
            x.dataX().real() * y.dataX().imag() + x.dataX().imag() * y.dataX().real()
        );
        Value <ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		__k.add_child(&x);
		__k.add_child(&y);

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(
				__k.get_grad().real() * y.dataX().real() - __k.get_grad().imag() * y.dataX().imag(),
				__k.get_grad().real() * y.dataX().imag() + __k.get_grad().imag() * y.dataX().real()
			);
			y.add_grad(
				__k.get_grad().real() * x.dataX().real() - __k.get_grad().imag() * x.dataX().imag(),
				__k.get_grad().real() * x.dataX().imag() + __k.get_grad().imag() * x.dataX().real()
			);
		});

		return __k;
    }

    Value<ResultType<T, U>> __k = x;
    __k *= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() * y.dataX());
        y.add_grad(__k.get_grad() * x.dataX());
    });

    return __k;
}


template <class T>
inline
Value <T>
operator* (const Value<T>& x, const T& y) {
    if constexpr (is_complex_v<T>) {
        T val(
            x.dataX().real() * y.real() - x.dataX().imag() * y.imag(),
            x.dataX().real() * y.imag() + x.dataX().imag() * y.real()
        );
        Value <T> __k = Value<T>(val);

		__k.add_child(&x);

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(
				__k.get_grad().real() * y.real() - __k.get_grad().imag() * y.imag(),
				__k.get_grad().real() * y.imag() + __k.get_grad().imag() * y.real()
			);
		});

        return __k;
    }

    Value<T> __k = x;
    __k *= y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() * y);
    });

    return __k;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator* (const Value<T>& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.dataX().real() * y.real() - x.dataX().imag() * y.imag(),
            x.dataX().real() * y.imag() + x.dataX().imag() * y.real()
        );
        Value <ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		__k.add_child(static_cast<Value<ResultType<T, U>>*>(&x));

		__k.set_backward([&x, &y, &__k]() {
			x.add_grad(
				__k.get_grad().real() * y.real() - __k.get_grad().imag() * y.imag(),
				__k.get_grad().real() * y.imag() + __k.get_grad().imag() * y.real()
			);
		});

		return __k;
    }

    Value<ResultType<T, U>> __k = x;
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
    if constexpr (is_complex_v<T>) {
        T val(
            x.real() * y.real() - x.imag() * y.imag(),
            x.real() * y.imag() + x.imag() * y.real()
        );
        return Value<T>(val);
    }

    Value<T> __k = x;
    __k *= y;
    return __k;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator* (const T& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        ResultType<T, U> val(
            x.real() * y.real() - x.imag() * y.imag(),
            x.real() * y.imag() + x.imag() * y.real()
        );
        return Value<ResultType<T, U>>(val);
    }

    Value<ResultType<T, U>> __k = x;
    __k *= y;
    return __k;
}


template <class T>
inline
Value <T>
operator/ (const Value<T>& x, const Value<T>& y) {
    if constexpr (is_complex_v<T>) {
        auto dr = y.dataX().real() * y.dataX().real() + y.dataX().imag() * y.dataX().imag();
        if (dr == 0) {
            throw std::invalid_argument("Division by zero");
        }

        T val(
            (x.dataX().real() * y.dataX().real() + x.dataX().imag() * y.dataX().imag()) / dr,
            (x.dataX().imag() * y.dataX().real() - x.dataX().real() * y.dataX().imag()) / dr
        );
        Value <T> __k = Value<T>(val);

		__k.add_child(&x);
		__k.add_child(&y);

        __k.set_backward([&x, &y, &__k]() {
            auto dr = y.dataX().real() * y.dataX().real() + y.dataX().imag() * y.dataX().imag();
            x.add_grad(
                (__k.get_grad().real() * y.dataX().real() + __k.get_grad().imag() * y.dataX().imag()) / dr,
                (__k.get_grad().real() * y.dataX().imag() - __k.get_grad().imag() * y.dataX().real()) / dr
            );
            y.add_grad(
                (-__k.get_grad().real() * x.dataX().imag() + __k.get_grad().imag() * x.dataX().real()) / dr,
                (-__k.get_grad().real() * x.dataX().real() - __k.get_grad().imag() * x.dataX().imag()) / dr
            );
        });

		return __k;
    }

    if (y.dataX() == 0) {
        throw std::invalid_argument("Division by zero");
    }

    using Type = std::conditional_t<std::is_integral_v<T>, double, T>;
    Type _k = static_cast<Type>(x.dataX()) / static_cast<Type>(y.dataX());

    Value<T> __k = Value<T>(static_cast<T>(_k));

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() / y.dataX());
        y.add_grad(-__k.get_grad() * x.dataX() / (y.dataX() * y.dataX()));
    });

    return __k;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator/ (const Value<T>& x, const Value<U>& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        auto dr = y.dataX().real() * y.dataX().real() + y.dataX().imag() * y.dataX().imag();
        if (dr == 0) {
            throw std::invalid_argument("Division by zero");
        }

        ResultType<T, U> val(
            (x.dataX().real() * y.dataX().real() + x.dataX().imag() * y.dataX().imag()) / dr,
            (x.dataX().imag() * y.dataX().real() - x.dataX().real() * y.dataX().imag()) / dr
        );
		Value<ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		__k.add_child(&x);
		__k.add_child(&y);

		__k.set_backward([&x, &y, &__k]() {
            auto dr = y.dataX().real() * y.dataX().real() + y.dataX().imag() * y.dataX().imag();
			x.add_grad(
				(__k.get_grad().real() * y.dataX().real() + __k.get_grad().imag() * y.dataX().imag()) / dr,
				(__k.get_grad().real() * y.dataX().imag() - __k.get_grad().imag() * y.dataX().real()) / dr
			);
			y.add_grad(
				(__k.get_grad().real() * x.dataX().imag() - __k.get_grad().imag() * x.dataX().real()) / dr,
				(__k.get_grad().real() * x.dataX().real() - __k.get_grad().imag() * x.dataX().imag()) / dr
			);
		});

        return Value<ResultType<T, U>>(val);
    }

    if (y.dataX() == 0) {
        throw std::invalid_argument("Division by zero");
    }

    Value<ResultType<T, U>> __k = x;
    __k /= y;

    __k.add_child(&x);
    __k.add_child(&y);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() / y.dataX());
        y.add_grad(-__k.get_grad() * x.dataX() / (y.dataX() * y.dataX()));
    });

    return __k;
}


template <class T>
inline
Value <T>
operator/ (const Value<T>& x, const T& y) {
    if constexpr (is_complex_v<T>) {
        auto dr = y.real() * y.real() + y.imag() * y.imag();
        if (dr == 0) {
            throw std::invalid_argument("Division by zero");
        }

        T val(
            (x.dataX().real() * y.real() + x.dataX().imag() * y.imag()) / dr,
            (x.dataX().imag() * y.real() + x.dataX().real() * y.imag()) / dr
        );
        Value <T> __k = Value<T>(val);

		__k.add_child(&x);

		__k.set_backward([&x, &y, &__k]() {
            auto dr = y.real() * y.real() + y.imag() * y.imag();
			x.add_grad(
				(__k.get_grad().real() * y.real() + __k.get_grad().imag() * y.imag()) / dr,
				(__k.get_grad().real() * y.imag() - __k.get_grad().imag() * y.real()) / dr
			);
		});

        return __k;
    }

    if (y == 0) {
        throw std::invalid_argument("Division by zero");
    }

    //Value<T> __k = x;
    //__k /= y;

    using Type = std::conditional_t<std::is_integral_v<T>, double, T>;
    Type _k = static_cast<Type>(x.dataX()) / static_cast<Type>(y);

    Value<T> __k = Value<T>(static_cast<T>(_k));

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() / y);
    });

    return __k;
}


template <class T, class U>
inline
Value <ResultType<T, U>>
operator/ (const Value<T>& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        auto dr = y.real() * y.real() + y.imag() * y.imag();
        if (dr == 0) {
            throw std::invalid_argument("Division by zero");
        }

        ResultType<T, U> val(
            (x.dataX().real() * y.real() + x.dataX().imag() * y.imag()) / dr,
            (x.dataX().imag() * y.real() + x.dataX().real() * y.imag()) / dr
        );
        Value<ResultType<T, U>> __k = Value<ResultType<T, U>>(val);

		__k.add_child(static_cast<Value<ResultType<T, U>>*>(&x));

		__k.set_backward([&x, &y, &__k]() {
            auto dr = y.real() * y.real() + y.imag() * y.imag();
			x.add_grad(
				(__k.get_grad().real() * y.real() + __k.get_grad().imag() * y.imag()) / dr,
				(__k.get_grad().real() * y.imag() - __k.get_grad().imag() * y.real()) / dr
			);
		});

		return __k;
    }

    if (y == 0) {
        throw std::invalid_argument("Division by zero");
    }

    Value<ResultType<T, U>> __k = x;
    __k /= y;

    __k.add_child(&x);

    __k.set_backward([&x, &y, &__k]() {
        x.add_grad(__k.get_grad() / y);
    });

    return __k;
}


template <class T>
inline
Value <T>
operator/ (const T& x, const T& y) {
    if constexpr (is_complex_v<T>) {
        auto dr = y.real() * y.real() + y.imag() * y.imag();
        if (dr == 0) {
            throw std::invalid_argument("Division by zero");
        }

        T val(
            (x.real() * y.real() + x.imag() * y.imag()) / dr,
            (x.imag() * y.real() + x.real() * y.imag()) / dr
        );
        return Value<T>(val);
    }

    if (y == 0) {
        throw std::invalid_argument("Division by zero");
    }

    //Value<T> __k = x;
    //__k /= y;

    using Type = std::conditional_t<std::is_integral_v<T>, double, T>;
    Type _k = static_cast<Type>(x) / static_cast<Type>(y);

    Value<T> __k = Value<T>(static_cast<T>(_k));

    return __k;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
operator/ (const T& x, const U& y) {
    if constexpr (is_complex_v<T> && is_complex_v<U>) {
        auto dr = y.real() * y.real() + y.imag() * y.imag();
        if (dr == 0) {
            throw std::invalid_argument("Division by zero");
        }

        ResultType<T, U> val(
            (x.real() * y.real() + x.imag() * y.imag()) / dr,
            (x.imag() * y.real() + x.real() * y.imag()) / dr
        );
        return Value<ResultType<T, U>>(val);
    }

    if (y == 0) {
        throw std::invalid_argument("Division by zero");
    }

    Value<ResultType<T, U>> __k = x;
    __k /= y;
    return __k;
}


template <class T>
inline
Value <T>
operator-(const Value<T>& _x) {
    if constexpr (is_complex_v<T>) {
        Value<T> __k = Value<T>(-_x.dataX().real(), -_x.dataX().imag());

        __k.add_child(&_x);

        __k.set_backward([&_x, &__k]() {
            _x.add_grad(-__k.get_grad());
        });

        return __k;
    } else {
        Value<T> __k = Value<T>(-_x.dataX());

        __k.add_child(&_x);

        __k.set_backward([&_x, &__k]() {
            _x.add_grad(-__k.get_grad());
        });

        return __k;
    }
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

template <class T, class U>
inline
Value <ResultType<T, U>>
add(const Value<T>& _x, const Value<U>& _y) {
    return _x + _y;
}

template <class T>
inline
Value <T>
add(const Value<T>& _x, const T& _y) {
    return _x + _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
add(const Value<T>& _x, const U& _y) {
    return _x + _y;
}

template <class T>
inline
Value <T>
add(const T& _x, const T& _y) {
    return _x + _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
add(const T& _x, const U& _y) {
    return _x + _y;
}

template <class T>
inline
std::vector<Value<T>>
add(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] + _y[i]);
    }
    return __k;
}

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
add(const std::vector<Value<T>>& _x, const std::vector<Value<U>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<ResultType<T, U>>> __k;
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


template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
add(const std::vector<Value<T>>& _x, const U& _y) {
    std::vector<Value<ResultType<T, U>>> __k;
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(add(_x[i], _y[i]));
    }
    return __k;
}


template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
add(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<U>>>& _y
) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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


template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
add(
    const std::vector<std::vector<Value<T>>>& _x,
    const U& _y
) {
    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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

template <class T, class U>
inline
Value <ResultType<T, U>>
sub(const Value<T>& _x, const Value<U>& _y) {
    return _x - _y;
}

template <class T>
inline
Value <T>
sub(const Value<T>& _x, const T& _y) {
    return _x - _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
sub(const Value<T>& _x, const U& _y) {
    return _x - _y;
}

template <class T>
inline
Value <T>
sub(const T& _x, const T& _y) {
    return _x - _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
sub(const T& _x, const U& _y) {
    return _x - _y;
}


template <class T>
inline
std::vector<Value<T>>
sub(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] - _y[i]);
    }
    return __k;
}


template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
sub(const std::vector<Value<T>>& _x, const std::vector<Value<U>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<ResultType<T, U>>> __k;
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


template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
sub(const std::vector<Value<T>>& _x, const U& _y) {
    std::vector<Value<ResultType<T, U>>> __k;
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(sub(_x[i], _y[i]));
    }
    return __k;
}


template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
sub(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<U>>>& _y
) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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


template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
sub(
    const std::vector<std::vector<Value<T>>>& _x,
    const U& _y
) {
    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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

template <class T, class U>
inline
Value <ResultType<T, U>>
rsub(const Value<T>& _x, const Value<U>& _y) {
    return _y - _x;
}

template <class T>
inline
Value <T>
rsub(const Value<T>& _x, const T& _y) {
    return _y - _x.dataX();
}

template <class T, class U>
inline
Value <ResultType<T, U>>
rsub(const Value<T>& _x, const U& _y) {
    return _y - _x.dataX();
}

template <class T>
inline
Value <T>
rsub(const T& _x, const T& _y) {
    return _y - _x;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
rsub(const T& _x, const U& _y) {
    return _y - _x;
}

template <class T>
inline
std::vector<Value<T>>
rsub(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_y[i] - _x[i]);
    }
    return __k;
}

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
rsub(const std::vector<Value<T>>& _x, const std::vector<Value<U>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<ResultType<T, U>>> __k;
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

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
rsub(const std::vector<Value<T>>& _x, const U& _y) {
    std::vector<Value<ResultType<T, U>>> __k;
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(rsub(_x[i], _y[i]));
    }
    return __k;
}

template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
rsub(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<U>>>& _y
) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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


template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
rsub(
    const std::vector<std::vector<Value<T>>>& _x,
    const U& _y
) {
    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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

template <class T, class U>
inline
Value <ResultType<T, U>>
mul(const Value<T>& _x, const Value<U>& _y) {
    return _x * _y;
}

template <class T>
inline
Value <T>
mul(const Value<T>& _x, const T& _y) {
    return _x * _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
mul(const Value<T>& _x, const U& _y) {
    return _x * _y;
}

template <class T>
inline
Value <T>
mul(const T& _x, const T& _y) {
    return _x * _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
mul(const T& _x, const U& _y) {
    return _x * _y;
}

template <class T>
inline
std::vector<Value<T>>
mul(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        //__k.push_back(mul(_x[i], _y[i]));
        if constexpr (is_complex_v<T>) {
            T val(
                _x[i].dataX().real() * _y[i].dataX().real() - _x[i].dataX().imag() * _y[i].dataX().imag(),
                _x[i].dataX().real() * _y[i].dataX().imag() + _x[i].dataX().imag() * _y[i].dataX().real()
            );
            __k.push_back(Value<T>(val));
        } else {
            __k.push_back(_x[i] * _y[i]);
        }
    }
    return __k;
}

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
mul(const std::vector<Value<T>>& _x, const std::vector<Value<U>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<ResultType<T, U>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(mul(_x[i], _y[i]));
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
        __k.push_back(mul(_x[i], _y));
    }
    return __k;
}

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
mul(const std::vector<Value<T>>& _x, const U& _y) {
    std::vector<Value<ResultType<T, U>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(mul(_x[i], _y));
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(mul(_x[i], _y[i]));
    }
    return __k;
}

template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
mul(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<U>>>& _y
) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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

template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
mul(
    const std::vector<std::vector<Value<T>>>& _x,
    const U& _y
) {
    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
    __k.reserve(_x.size());
    for (int i = 0; i < _x.size(); ++i) {
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

template <class T, class U>
inline
Value <ResultType<T, U>>
div(const Value<T>& _x, const Value<U>& _y) {
    return _x / _y;
}

template <class T>
inline
Value <T>
div(const Value<T>& _x, const T& _y) {
    return _x / _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
div(const Value<T>& _x, const U& _y) {
    return _x / _y;
}

template <class T>
inline
Value <T>
div(const T& _x, const T& _y) {
    return _x / _y;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
div(const T& _x, const U& _y) {
    return _x / _y;
}

template <class T>
inline
std::vector<Value<T>>
div(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_x[i] / _y[i]);
    }
    return __k;
}

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
div(const std::vector<Value<T>>& _x, const std::vector<Value<U>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<ResultType<T, U>>> __k;
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

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
div(const std::vector<Value<T>>& _x, const U& _y) {
    std::vector<Value<ResultType<T, U>>> __k;
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(div(_x[i], _y[i]));
    }
    return __k;
}

template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
div(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<U>>>& _y
) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

	std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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

template <class T, class U>
inline
std::vector<std::vector<Value<ResultType<T, U>>>>
div(
    const std::vector<std::vector<Value<T>>>& _x,
    const U& _y
) {
    std::vector<std::vector<Value<ResultType<T, U>>>> __k;
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

template <class T, class U>
inline
Value <ResultType<T, U>>
rdiv(const Value<T>& _x, const Value<U>& _y) {
    return _y / _x;
}

template <class T>
inline
Value <T>
rdiv(const Value<T>& _x, const T& _y) {
    return _y / _x.dataX();
}

template <class T, class U>
inline
Value <ResultType<T, U>>
rdiv(const Value<T>& _x, const U& _y) {
    return _y / _x.dataX();
}

template <class T>
inline
Value <T>
rdiv(const T& _x, const T& _y) {
    return _y / _x;
}

template <class T, class U>
inline
Value <ResultType<T, U>>
rdiv(const T& _x, const U& _y) {
    return _y / _x;
}

template <class T>
inline
std::vector<Value<T>>
rdiv(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(_y[i] / _x[i]);
    }
    return __k;
}

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
rdiv(const std::vector<Value<T>>& _x, const std::vector<Value<U>>& _y) {
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<Value<ResultType<T, U>>> __k;
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

template <class T, class U>
inline
std::vector<Value<ResultType<T, U>>>
rdiv(const std::vector<Value<T>>& _x, const U& _y) {
    std::vector<Value<ResultType<T, U>>> __k;
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
    Value<T> __k = Value<T>(std::pow(_x.dataX(), _y.dataX()));

	__k.add_child(&_x);
	__k.add_child(&_y);

	__k.set_backward([&_x, &_y, &__k]() {
		_x.add_grad(__k.get_grad() * _y.dataX() * std::pow(_x.dataX(), _y.dataX() - 1));
		_y.add_grad(__k.get_grad() * std::pow(_x.dataX(), _y.dataX()) * std::log(_x.dataX()));
	});

	return __k;
}


template <class T>
inline
Value <T>
pow(const Value<T>& _x, const T& _y) {
    Value<T> __k = Value<T>(std::pow(_x.dataX(), _y));

	__k.add_child(&_x);

	__k.set_backward([&_x, &_y, &__k]() {
		_x.add_grad(__k.get_grad() * _y * std::pow(_x.dataX(), _y - 1));
	});

	return __k;
}

template <class T>
inline
Value <T>
pow(const T& _x, const T& _y) {
    Value<T> __k;
    __k = std::pow(_x, _y);
    return Value<T>(__k);
}

/*
template <class T>
inline
std::vector<Value<T>>
pow(const std::vector<Value<T>>& _x, const std::vector<Value<T>>& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(pow(_x[i], _y[i]));
    }
    return __k;
}

template <class T>
inline
std::vector<Value<T>>
pow(const std::vector<Value<T>>& _x, const T& _y) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(pow(_x[i], _y));
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<Value<T>>>
pow(
    const std::vector<std::vector<Value<T>>>& _x,
    const std::vector<std::vector<Value<T>>>& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(pow(_x[i], _y[i]));
    }
    return __k;
}

template <class T>
inline
std::vector<std::vector<Value<T>>>
pow(
    const std::vector<std::vector<Value<T>>>& _x,
    const T& _y
) {
    std::vector<std::vector<Value<T>>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(pow(_x[i], _y));
    }
    return __k;
}
*/


// neg

template <class T>
inline
Value <T>
neg(const Value<T>& _x) {
    if constexpr (is_complex_v<T>) {
        T _k(-_x.dataX().real(), -_x.dataX().imag());
        Value<T> __k = Value<T>(_k);

        __k.add_child(&_x);

        __k.set_backward([&_x, &__k]() {
            _x.add_grad(-__k.get_grad().real(), -__k.get_grad().imag());
        });

        return __k;
    } else {
        Value<T> __k = Value<T>(-_x.dataX());

        __k.add_child(&_x);

        __k.set_backward([&_x, &__k]() {
            _x.add_grad(-__k.get_grad());
        });

        return __k;
    }
}


template <class T>
inline
Value <T>
neg(const T& _x) {
    if constexpr (is_complex_v<T>) {
        T _k(-_x.real(), -_x.imag());
        Value<T> __k = Value<T>(_k);
        return __k;
    } else {
        Value<T> __k = Value<T>(-_x);
        return __k;
    }
}


template <class T>
inline
std::vector<Value<T>>
neg(const std::vector<Value<T>>& _x) {
    std::vector<Value<T>> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        if constexpr (is_complex_v<T>) {
            T val(-_x[i].dataX().real(), -_x[i].dataX().imag());
            __k.push_back(Value<T>(val));
        } else {
            __k.push_back(Value<T>(-_x[i].dataX()));
        }
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


// lt

template <class T>
inline
bool
lt(const Value<T>& _x, const Value<T>& _y) {
    return _x.dataX() < _y.dataX();
}

template <class T>
inline
bool
lt(const Value<T>& _x, const T& _y) {
    return _x.dataX() < _y;
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<bool> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(lt(_x[i], _y[i]));
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
        __k.push_back(lt(_x[i], _y));
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

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
    return _x.dataX() > _y.dataX();
}

template <class T>
inline
bool
gt(const Value<T>& _x, const T& _y) {
    return _x.dataX() > _y;
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    std::vector<bool> __k;
    __k.reserve(_x.size());
    for (size_t i = 0; i < _x.size(); ++i) {
        __k.push_back(gt(_x[i], _y[i]));
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
        __k.push_back(gt(_x[i], _y));
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
    if (_x.size() != _y.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

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
    Value<T> __k;
    if (_x.dataX() < 0.0) {
        __k = Value<T>(0.0);
    } else {
        __k = Value<T>(_x.dataX());
    }

    __k.add_child(&_x);

    __k.set_backward([&_x, &__k]() {
        if (_x.dataX() < 0.0) {
            _x.add_grad(0.0);
        } else {
            _x.add_grad(__k.get_grad());
        }
    });

    return __k;
}

template <class T>
inline
Value <T>
relu(const T& _x) {
    Value <T> __k;
    if (_x < 0.0) {
        __k = Value<T>(0.0);
    } else {
        __k = Value<T>(_x);
    }
    __k.set_grad(0.0);
    return __k;
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
