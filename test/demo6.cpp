#include <iostream>

#include "../src/engine.h"

using namespace ptMgrad;


int main() {
    Value<double> a = 2.0;
    Value<double> b = 3.0;
    Value<double> c = 4.0;
    Value<double> d = 5.0;

    Value<double> e = a + b * c - d;
    e.backward();

    std::cout << e.dataX() << std::endl;
    std::cout << e.gradX() << std::endl;

    return 0;
}
