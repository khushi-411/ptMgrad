#include <iostream>

#include "src/engine.h"
#include "src/complex.h"


int main() {
    using namespace ptMgrad;

    Value<float> a = 2.0f;
    Value<float> b = -3.0f;

    Value<float> c = a + b;

    std::cout << a.dataX() << std::endl;
    std::cout << b.dataX() << std::endl;
    std::cout << c.dataX() << std::endl;

    c.backward();

    std::cout << c.gradX() << std::endl;
    std::cout << a.gradX() << std::endl;
    std::cout << b.gradX() << std::endl;
}
