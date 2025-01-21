// create a neural net demonstration

#include <iostream>
#include <vector>
#include <random>

#include "src/engine.h"
#include "src/nn.h"


using namespace ptMgrad;


int main() {
    // initialize the model
    MLP<double> model(2, {3, 1});

    // create a dataset
    std::vector<std::vector<Value<double>>> x = {
        {Value<double>(0.0), Value<double>(0.0)},
        {Value<double>(0.0), Value<double>(1.0)},
        {Value<double>(1.0), Value<double>(0.0)},
        {Value<double>(1.0), Value<double>(1.0)}
    };

    std::vector<Value<double>> y = {
        Value<double>(0.0),
        Value<double>(1.0),
        Value<double>(1.0),
        Value<double>(0.0)
    };

    // train the model
    for (int i = 0; i < 100; i++) {
        for (size_t j = 0; j < x.size(); j++) {
            auto ypred = model(x[j]);
            auto loss = (ypred[0] - y[j]).pow(2);
            loss.backward();
            model.zero_grad();
        }
    }

    // print the model
    model.print();

    // print the predictions
    for (size_t i = 0; i < x.size(); i++) {
        auto ypred = model(x[i]);
        std::cout << "x: " << x[i][0].dataX() << ", " << x[i][1].dataX() << " -> y: " << ypred[0].dataX() << std::endl;
    }

    std::cout << "Done!" << std::endl;

    return 0;
}