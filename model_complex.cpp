#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#include "nn.h"

using namespace ptMgrad;

int main() {
    std::srand(0);

    using C = complex<double>;
    using V = Value<C>;

    // complex inputs
    std::vector<std::vector<V>> xs = {
        {V(C(1.0,  0.5)), V(C(-0.5, 1.0))},
        {V(C(0.2, -1.0)), V(C( 1.5, 0.3))},
        {V(C(-1.0, 0.0)), V(C( 0.7, -0.7))},
    };

    // Model — complex MLP: 2 -> hidden(4, CReLU) -> output(1, linear)
    MLP<C> model(2, {4, 1});

    model.print();
    std::cout << "Number of parameters: " << model.parameters().size() << "\n\n";

    // forward pass
    std::cout << "--- Forward pass (complex outputs) ---\n";
    for (size_t i = 0; i < xs.size(); ++i) {
        V pred = model(xs[i])[0];
        std::cout << "x=(" << xs[i][0] << ", " << xs[i][1] << ")"
                  << "  ->  pred=" << pred << "\n";
    }

    // backward pass
    std::cout << "\n--- Backward pass on first sample ---\n";
    model.zero_grad();
    V out = model(xs[0])[0];
    out.backward();

    std::cout << "output = " << out << "\n";
    std::cout << "complex gradients w.r.t. the first 5 parameters:\n";

    auto params = model.parameters();
    for (size_t i = 0; i < params.size() && i < 5; ++i) {
        std::cout << "  param[" << i << "]  value=" << params[i]->dataX()
                  << "  grad=" << params[i]->get_grad() << "\n";
    }

    std::cout << "\nComplex autograd passes.\n";
    return 0;
}
