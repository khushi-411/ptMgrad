#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "nn.h"

using namespace ptMgrad;

int main() {
    std::srand(0);

    // dataset
    using V = Value<double>;

    std::vector<std::vector<V>> xs = {
        {V(0.0), V(0.0)},
        {V(0.0), V(1.0)},
        {V(1.0), V(0.0)},
        {V(1.0), V(1.0)},
    };
    std::vector<double> ys = {0.0, 1.0, 1.0, 0.0};

    // Model — MLP: 2 inputs -> hidden(4, ReLU) -> output(1, linear)
    MLP<double> model(2, {4, 1});

    model.print();
    std::cout << "Number of parameters: " << model.parameters().size() << "\n\n";

    // training
    const double lr     = 0.1;
    const int    epochs = 200;

    for (int epoch = 0; epoch < epochs; ++epoch) {

        // forward pass: accumulate MSE loss
        // Use operator+ (not +=) so every addition is a graph node.
        V loss(0.0);
        for (size_t i = 0; i < xs.size(); ++i) {
            V pred = model(xs[i])[0];
            V diff = pred - V(ys[i]);
            V sq   = pow(diff, 2.0);
            loss   = loss + sq;
        }
        // mean
        V mse = loss * V(1.0 / static_cast<double>(xs.size()));

        // backward pass
        model.zero_grad();
        mse.backward();

        // SGD parameter update
        for (auto* p : model.parameters()) {
            double updated = p->dataX() - lr * p->get_grad();
            *p = updated;
        }

        if (epoch % 20 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(3) << epoch
                      << "  MSE = " << mse << "\n";
        }
    }

    // predictions
    std::cout << "\n--- Predictions after training ---\n";
    for (size_t i = 0; i < xs.size(); ++i) {
        V pred = model(xs[i])[0];
        std::cout << "x=(" << xs[i][0] << "," << xs[i][1] << ")"
                  << "  pred=" << pred
                  << "  target=" << ys[i] << "\n";
    }

    return 0;
}
