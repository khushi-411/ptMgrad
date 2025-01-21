#pragma once

#include <vector>
#include <iostream>
#include <random>

#include "engine.h"


namespace ptMgrad {

template <typename T>
class Module {
public:
    virtual std::vector<Value<T>*> parameters() = 0;
    virtual void zero_grad() {
        for (auto* p : parameters()) {
            p->grad = 0.0;
        }
    }
};


template <typename T>
class Neuron : public Module<T> {
private:
    std::vector<Value<T>> w;
    Value<T> b;
    bool nonlin;

public:
    Neuron(int nin, bool nonlin = true) : w(nin), b(0), nonlin(nonlin) {
        for (auto& wi : w) {
            wi = Value<T>(rand() / RAND_MAX * 2.0 - 1.0);
        }
    }

    Value<T> operator()(const std::vector<Value<T>>& x) {
        Value<T> act = b;
        for (size_t i = 0; i < w.size(); ++i) {
            act += w[i] * x[i];
        }
        return nonlin ? act.relu() : act; 
    }

    std::vector<Value<T>*> parameters() override {
        std::vector<Value<T>*> params;
        for (auto& wi : w) {
            params.push_back(&wi);
        }
        params.push_back(&b);
        return params;
    }

    void print() {
        std::cout << (nonlin ? "ReLU" : "Linear") << "Neuron(" << w.size() << ")\n";
    }
};


template <typename T>
class Layer : public Module<T> {
private:
    std::vector<Neuron<T>> neurons;
    bool nonlin;
    int nin;
    int nout;
    std::vector<Value<T>*> params;

public:
    Layer(int nin, int nout, bool nonlin = true) : nin(nin), nout(nout), nonlin(nonlin) {
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(Neuron<T>(nin, nonlin));
        }
    }

    std::vector<Value<T>> operator()(const std::vector<Value<T>>& x) {
        std::vector<Value<T>> out;
        for (auto& n : neurons) {
            out.push_back(n(x));
        }
        return out;
    }

    std::vector<Value<T>*> parameters() override {
        std::vector<Value<T>*> params;
        for (auto& n : neurons) {
            for (auto* p : n.parameters()) {
                params.push_back(p);
            }
        }
        return params;
    }

    void print() {
        std::cout << "Layer of [";
        for (auto& n : neurons) {
            n.print();
        }
        std::cout << "]\n";
    }
};


template <typename T>
class MLP : public Module<T> {
private:
    std::vector<Layer<T>> layers;
    std::vector<Value<T>*> params;
    int nin;
    std::vector<int> nouts;

public:
    MLP(int nin, const std::vector<int>& nouts) : nin(nin), nouts(nouts) {
        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.push_back(Layer<T>(nin, nouts[i], i != nouts.size() - 1));
            nin = nouts[i];
        }
    }

    std::vector<Value<T>> operator()(const std::vector<Value<T>>& x) {
        std::vector<Value<T>> out = x;
        for (auto& layer : layers) {
            out = layer(out);
        }
        return out;
    }

    std::vector<Value<T>*> parameters() override {
        std::vector<Value<T>*> params;
        for (auto& layer : layers) {
            for (auto* p : layer.parameters()) {
                params.push_back(p);
            }
        }
        return params;
    }

    void print() {
        std::cout << "MLP of [";
        for (auto& layer : layers) {
            layer.print();
        }
        std::cout << "]\n";
    }
};

}