#ifndef ARRAY_H
#define ARRAY_H

#include <vector>

#include "engine.h"


namespace ptMgrad {

// TODO: add initializer list

// construct Array class
template <class T>
class Array {
private:
    T* data;
    size_t capacity;
    size_t _size;

    void resize(size_t new_capacity) {
        T* new_data = new T[new_capacity];
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = data[i];
        }
        delete[] data;
        data = new_data;
        capacity = new_capacity;
    }

public:
    Array() : data(nullptr), capacity(0), _size(0) {}

    explicit Array(size_t n) : capacity(n), _size(0) {
        data = new T[capacity];
    }

    ~Array() {
        delete[] data;
    }

    Array(const Array& other) : capacity(other.capacity), _size(other._size) {
        data = new T[capacity];
        for (size_t i = 0; i < _size; i++) {
            data[i] = other.data[i];
        }
    }

    Array& operator=(const Array& other) {
        if (this != &other) {
            delete[] data;
            capacity = other.capacity;
            _size = other._size;
            data = new T[capacity];
            for (size_t i = 0; i < _size; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    Array(Array&& other) noexcept : data(other.data), capacity(other.capacity), _size(other._size) {
        other.data = nullptr;
        other.capacity = 0;
        other._size = 0;
    }

    Array& operator=(Array&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            capacity = other.capacity;
            _size = other._size;
            other.data = nullptr;
            other.capacity = 0;
            other._size = 0;
        }
        return *this;
    }

    // why const needed?
    T& operator[](size_t index) const {
        if (index >= _size) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    T& at(size_t index) {
        if (index >= _size) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    size_t size() const {
        return _size;
    }

    size_t get_capacity() const {
        return capacity;
    }

    bool empty() const {
        return _size == 0;
    }

    void push_back(const T& value) {
        if (_size >= capacity) {
            resize(capacity == 0 ? 1 : capacity * 2);
        }
        data[_size++] = value;
    }

    void pop_back() {
        if (_size > 0) {
            _size--;
        }
    }

    void clear() {
        _size = 0;
    }

    T* begin() {
        return data;
    }

    T* end() {
        return data + _size;
    }

    const T* begin() const {
        return data;
    }

    const T* end() const {
        return data + _size;
    }
};


template <typename T>
Array<Value<T>>
operator+ (const Array<Value<T>>& x, const Array<Value<T>>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }

    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] + y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator+ (const Array<Value<T>>& x, const T& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] + y);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator+ (const T& x, const Array<Value<T>>& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < y.size(); ++i) {
        __k.push_back(x + y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator- (const Array<Value<T>>& x, const Array<Value<T>>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }

    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] - y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator- (const Array<Value<T>>& x, const T& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] - y);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator- (const T& x, const Array<Value<T>>& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < y.size(); ++i) {
        __k.push_back(x - y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator* (const Array<Value<T>>& x, const Array<Value<T>>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }

    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] * y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator* (const Array<Value<T>>& x, const T& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] * y);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator* (const T& x, const Array<Value<T>>& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < y.size(); ++i) {
        __k.push_back(x * y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator/ (const Array<Value<T>>& x, const Array<Value<T>>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }

    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] / y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator/ (const Array<Value<T>>& x, const T& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(x[i] / y);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator/ (const T& x, const Array<Value<T>>& y) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < y.size(); ++i) {
        __k.push_back(x / y[i]);
    }
    return __k;
}

template <typename T>
Array<Value<T>>
operator- (const Array<Value<T>>& x) {
    Array<Value<T>> __k;
    for (size_t i = 0; i < x.size(); ++i) {
        __k.push_back(-x[i]);
    }
    return __k;
}
/*
template <typename T>
Array<Value<T>>
add(const Array<Value<T>>& _x, const Array<Value<T>>& _y) {
    return _x + _y;
}
*/
}

#endif