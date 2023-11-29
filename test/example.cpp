#include "../src/engine.h"

int main() {
    ptMgrad::Value<double> a, b;
    a = 2.0;
    b = 3.0;
    auto c = 4.0;
    auto d = 5.0;
    ptMgrad::Value<double> res = ptMgrad::add(a, b);
    std::cout << res << std::endl;
    // TODO: for scalar values why we need val::add(), instead of just add() like in others
    // val::Value<double> res = val::add(c, c);
    return 0;
}
