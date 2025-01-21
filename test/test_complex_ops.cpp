#include <iostream>

#include <gtest/gtest.h>

#include "../src/engine.h"


using namespace ptMgrad;

/*
// for complex operations
TEST(ValueTest, BasicComplexFloatOperations) {
    // Value<complex<float>> a.real() = 2.0f;
    // Value<complex<float>> a.imag() = 1.0f;
    // Value<complex<float>> b.real() = 3.0f;
    // Value<complex<float>> b.imag() = 2.0f;

    // EXPECT_EQ(real(a).dataX(), 2.0f);
    // EXPECT_EQ(real(b).dataX(), 3.0f);

    Value<complex<float>> a(complex<complex<float>>(1.0f, 2.0f));
    Value<complex<float>> b(complex<complex<float>>(3.0f, 4.0f));

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.dataX().real(), 3.0f);
    EXPECT_EQ(b.dataX().imag(), 4.0f);

    EXPECT_EQ(a + b, complex<complex<float>>(4.0f, 6.0f));
}
*/