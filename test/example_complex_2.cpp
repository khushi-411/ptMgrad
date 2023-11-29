#include "../src/engine.h"

// TODO: This does not show correct results

int main() {
    ptMgrad::Value<ptMgrad::complex<double>> a(ptMgrad::Value<ptMgrad::complex<double>>(1.0, 2.0));
    ptMgrad::Value<ptMgrad::complex<double>> b(ptMgrad::Value<ptMgrad::complex<double>>(3.0, 4.0));
    ptMgrad::Value<ptMgrad::complex<double>> res = ptMgrad::add(a, b);
    std::cout << res << std::endl;
    return 0;
}
