#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "engine.h"
#include "Array.h"

using namespace ptMgrad;
using V = Value<double>;

// dot product: Array<V> · Array<V> → scalar V
static V dot(const Array<V>& a, const Array<V>& b) {
    Array<V> ab = a * b;
    V s(0.0);
    for (size_t i = 0; i < ab.size(); ++i)
        s = s + ab[i];
    return s;
}

int main() {
    std::srand(0);

    // dataset
    Array<Array<V>> xs = {
        { V(0.0), V(0.0) },
        { V(0.0), V(1.0) },
        { V(1.0), V(0.0) },
        { V(1.0), V(1.0) },
    };
    Array<double> ys = { 0.0, 1.0, 1.0, 0.0 };

    // 2.  Model — 2 → 4 (ReLU) → 1 (linear)
    //
    //   W1[j]  = Array<V> of length 2   (weights for hidden neuron j)
    //   b1[j]  = V                      (bias for hidden neuron j)
    //   w2     = Array<V> of length 4   (output weights)
    //   b2     = V                      (output bias)
    const int NIN = 2, NHID = 4;

    auto randf = []() {
        return static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
    };

    Array<Array<V>> W1;
    Array<V> b1;
    for (int j = 0; j < NHID; ++j) {
        W1.push_back({ V(randf()), V(randf()) });
        b1.push_back(V(0.0));
    }

    Array<V> w2;
    V b2(0.0);
    for (int j = 0; j < NHID; ++j)
        w2.push_back(V(randf()));

    const int nparams = NHID * (NIN + 1) + NHID + 1;
    std::cout << "MLP 2 -> 4(ReLU) -> 1  |  parameters: " << nparams << "\n\n";

    // training
    const double lr = 0.1;
    const int epochs = 400;

    for (int epoch = 0; epoch < epochs; ++epoch) {

        // zero gradients
        for (int j = 0; j < NHID; ++j) {
            for (int i = 0; i < NIN; ++i) W1[j][i].zero_grad();
            b1[j].zero_grad();
            w2[j].zero_grad();
        }
        b2.zero_grad();

        // forward pass
        V loss(0.0);
        for (size_t s = 0; s < xs.size(); ++s) {

            // hidden layer
            Array<V> h;
            for (int j = 0; j < NHID; ++j)
                h.push_back(relu(dot(W1[j], xs[s]) + b1[j]));

            // output layer
            V pred = dot(w2, h) + b2;
            V diff = pred - V(ys[s]);
            loss   = loss + pow(diff, 2.0);
        }
        V mse = loss * V(1.0 / xs.size());

        // backward
        mse.backward();

        // SGD update
        for (int j = 0; j < NHID; ++j) {
            for (int i = 0; i < NIN; ++i)
                W1[j][i] = W1[j][i].dataX() - lr * W1[j][i].gradX();
            b1[j] = b1[j].dataX() - lr * b1[j].gradX();
            w2[j] = w2[j].dataX() - lr * w2[j].gradX();
        }
        b2 = b2.dataX() - lr * b2.gradX();

        if (epoch % 40 == 0 || epoch == epochs - 1)
            std::cout << "Epoch " << std::setw(3) << epoch
                      << "  MSE = " << mse.dataX() << "\n";
    }

    // predictions
    std::cout << "\n--- Predictions after training ---\n";
    for (size_t s = 0; s < xs.size(); ++s) {
        Array<V> h;
        for (int j = 0; j < NHID; ++j)
            h.push_back(relu(dot(W1[j], xs[s]) + b1[j]));
        V pred = dot(w2, h) + b2;
        std::cout << "x=(" << xs[s][0].dataX() << "," << xs[s][1].dataX() << ")"
                  << "  pred=" << std::fixed << std::setprecision(4) << pred.dataX()
                  << "  target=" << ys[s] << "\n";
    }

    return 0;
}
