#include <iostream>

#include "../src/engine.h"
#include "../src/complex.h"

using namespace ptMgrad;

int main() {
    Value<complex<double>> a(1.0, 2.0);
    complex<double> b(3.0, 4.0);

    std::cout << (a / b).dataX() << std::endl;

    return 0;
}
