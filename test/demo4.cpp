#include <iostream>

#include "../src/engine.h"
#include "../src/Array.h"


using namespace ptMgrad;


int main() {
    Value<float> a = 2.0;
    Value<float> b = 3.0;

    Value<double> c = a - b;

    std::cout << c.dataX() << std::endl;

    return 0;
}
