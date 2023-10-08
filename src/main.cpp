#include "engine.h"

int main() {
    val::Value<double> a, b;
    a = 2.0;
    b = 3.0;
    auto c = 4.0;
    auto d = 5.0;
    val::Value<double> res = val::pow(b, d);
    std::cout << res << std::endl;
    // TODO: for scalar values why we need val::add(), instead of just add() like in others
    // val::Value<double> res = val::add(c, c);
    return 0;
}
