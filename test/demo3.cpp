#include <iostream>

#include "../src/engine.h"
#include "../src/Array.h"


using namespace ptMgrad;


int main() {
    Array<Value<float>> a = {2.0f, 3.0f, 4.0f};
    Array<Value<double>> b = {2.0, 3.0, 4.0};

    Array<Value<double>> c = a + b;

    std::cout << c[0].dataX() << std::endl;
    std::cout << c[1].dataX() << std::endl;
    std::cout << c[2].dataX() << std::endl;

    return 0;
}