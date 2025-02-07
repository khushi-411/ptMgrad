#include <iostream>

#include "../src/engine.h"
#include "../src/Array.h"


using namespace ptMgrad;


int main() {
    Array<Value<double>> a;
    a.push_back(2.0);
    a.push_back(3.0);
    a.push_back(4.0);

    Array<Value<float>> b;
    b.push_back(2.0f);
    b.push_back(3.0f);
    b.push_back(4.0f);

    Array<Value<double>> c = a + b;

    std::cout << c[0].dataX() << std::endl;
    std::cout << c[1].dataX() << std::endl;
    std::cout << c[2].dataX() << std::endl;

    return 0;
}