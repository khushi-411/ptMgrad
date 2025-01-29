#include <iostream>

#include "../src/engine.h"
#include "../src/Array.h"


using namespace ptMgrad;


int main() {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    std::cout << (a + b).dataX() << " a = " << a.dataX() << " b = " << b.dataX() << std::endl;
    std::cout << (a - b).dataX() << " a = " << a.dataX() << " b = " << b.dataX() << std::endl;
    std::cout << (a * b).dataX() << " a = " << a.dataX() << " b = " << b.dataX() << std::endl;
    std::cout << (a / b).dataX() << " a = " << a.dataX() << " b = " << b.dataX() << std::endl;

    return 0;
}
