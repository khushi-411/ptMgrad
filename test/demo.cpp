#include <iostream>
#include <vector>

#include "engine.h"
#include "Array.h"


using namespace ptMgrad;


int main() {
    ptMgrad::Array<ptMgrad::Value<double>> a;
    a.push_back(2.0);
    a.push_back(3.0);
    a.push_back(4.0);

    std::cout << a[0].dataX() << std::endl;
    std::cout << a[1].dataX() << std::endl;
    std::cout << a[2].dataX() << std::endl;

    ptMgrad::Array<ptMgrad::Array<ptMgrad::Value<double>>> b;
    b.push_back(a);
    b.push_back(a);

    std::cout << "\n";
    std::cout << b[0][0].dataX() << std::endl;
    std::cout << b[0][1].dataX() << std::endl;
    std::cout << b[0][2].dataX() << std::endl;
    std::cout << b[1][0].dataX() << std::endl;
    std::cout << b[1][1].dataX() << std::endl;
    std::cout << b[1][2].dataX() << std::endl;

    ptMgrad::Array<ptMgrad::Array<ptMgrad::Value<double>>> c = b + b;

    std::cout << "\n";
    std::cout << c[0][0].dataX() << std::endl;
    std::cout << c[0][1].dataX() << std::endl;
    std::cout << c[0][2].dataX() << std::endl;
    std::cout << c[1][0].dataX() << std::endl;
    std::cout << c[1][1].dataX() << std::endl;
    std::cout << c[1][2].dataX() << std::endl;

    return 0;
}
