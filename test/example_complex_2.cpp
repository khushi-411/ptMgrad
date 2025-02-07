#include "../src/engine.h"
#include "../src/complex.h"


int main() {
    ptMgrad::Value<ptMgrad::complex<double>> a(ptMgrad::complex<double>(1.0, 2.0));
    ptMgrad::Value<ptMgrad::complex<double>> b(ptMgrad::complex<double>(3.0, 4.0));
    ptMgrad::Value<ptMgrad::complex<double>> res = ptMgrad::add(a, b);
    std::cout << res << std::endl;
    return 0;
}
