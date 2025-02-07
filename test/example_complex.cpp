#include "../src/engine.h"

int main() {
    ptMgrad::complex<double> a(1.0, 2.0), b(3.0, 4.0);
    ptMgrad::complex<double> res = ptMgrad::add(a, b);
    std::cout << res << std::endl;
    return 0;
}
