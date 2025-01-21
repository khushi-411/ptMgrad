#include <iostream>
#include <gtest/gtest.h>

#include "../src/engine.h"
#include "../src/complex.h"


using namespace ptMgrad;


// for float operations
TEST(ValueTest, BasicFloatOperations) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    EXPECT_EQ(a.dataX(), 2.0f);
    EXPECT_EQ(b.dataX(), 3.0f);

    EXPECT_EQ(a + b, 5.0f);
    EXPECT_EQ(a - b, -1.0f);
    EXPECT_EQ(a * b, 6.0f);
    EXPECT_EQ(a / b, 2.0f / 3.0f);
    
    // test addition
    Value<float> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0f);

    // test substraction
    Value<float> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0f);

    // test multiplication
    Value<float> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0f);

    // test division
    Value<float> f = a / b;
    EXPECT_EQ(f.dataX(), 2.0f / 3.0f);

    // using assignment operator
    
    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0f);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0f);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0f);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 2.0f / 3.0f);
}


// for double operations
TEST(ValueTest, BasicDoubleOperations) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    EXPECT_EQ(a.dataX(), 2.0);
    EXPECT_EQ(b.dataX(), 3.0);

    EXPECT_EQ(a + b, 5.0);
    EXPECT_EQ(a - b, -1.0);
    EXPECT_EQ(a * b, 6.0);
    EXPECT_EQ(a / b, 2.0 / 3.0);

    // test addition
    Value<double> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0);

    // test substraction
    Value<double> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0);

    // test multiplication
    Value<double> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0);

    // test division
    Value<double> f = a / b;
    EXPECT_EQ(f.dataX(), 2.0 / 3.0);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 2.0 / 3.0);
}


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


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
