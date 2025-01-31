#include <iostream>

#include <gtest/gtest.h>


#include "../src/engine.h"


using namespace ptMgrad;


/*
 * Test for unary operations:
 *     - neg
 * 
 * Type Conversions:
 *     - type1 = type2
 */


// neg

#define TEST_VALUE_NEG(TYPE, NAME)                    \
    TEST(ValueTest, Neg##NAME) {                      \
        Value<TYPE> a = 2.0;                          \
                                                      \
        Value<TYPE> b = ptMgrad::neg(a);              \
                                                      \
        EXPECT_EQ(b.dataX(), TYPE(-2.0));             \
    }

TEST_VALUE_NEG(float, Float)
TEST_VALUE_NEG(double, Double)
TEST_VALUE_NEG(int, Int)


TEST(ValueTest, ScalarNegFloat) {
    float a = 2.0f;

    Value<float> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX(), -2.0f);
}

TEST(ValueTest, ScalarNegDouble) {
    double a = 2.0;

    Value<double> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX(), -2.0);
}

TEST(ValueTest, VectorNeg) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};

    std::vector<Value<float>> b = ptMgrad::neg(a);

    EXPECT_EQ(b[0].dataX(), -2.0f);
    EXPECT_EQ(b[1].dataX(), -3.0f);
    EXPECT_EQ(b[2].dataX(), -4.0f);
}

TEST(ValueTest, MatrixNeg) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };

    std::vector<std::vector<Value<float>>> b = ptMgrad::neg(a);

    EXPECT_EQ(b[0][0].dataX(), -2.0f);
    EXPECT_EQ(b[0][1].dataX(), -3.0f);
    EXPECT_EQ(b[0][2].dataX(), -4.0f);
    EXPECT_EQ(b[1][0].dataX(), -5.0f);
    EXPECT_EQ(b[1][1].dataX(), 6.0f);
    EXPECT_EQ(b[1][2].dataX(), -7.0f);
}
/*
TEST(ValueTest, ComplexNeg) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));

    Value<complex<float>> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX().real(), -1.0f);
    EXPECT_EQ(b.dataX().imag(), -2.0f);
}

TEST(ValueTest, ComplexNegNeg) {
    Value<complex<float>> a(complex<float>(-1.0f, -2.0f));

    Value<complex<float>> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX().real(), 1.0f);
    EXPECT_EQ(b.dataX().imag(), 2.0f);
}
*/
TEST(ValueTest, ComplexNegScalar) {
    complex<float> a(1.0f, 2.0f);

    Value<complex<float>> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX().real(), -1.0f);
    EXPECT_EQ(b.dataX().imag(), -2.0f);
}
/*
TEST(ValueTest, ComplexNegVector) {
    std::vector<Value<complex<float>>> a = {
        Value<complex<float>>(complex<float>(1.0f, 2.0f)),
        Value<complex<float>>(complex<float>(3.0f, 4.0f)),
        Value<complex<float>>(complex<float>(5.0f, 6.0f))
    };

    std::vector<Value<complex<float>>> b = ptMgrad::neg(a);

    EXPECT_EQ(b[0].dataX().real(), -1.0f);
    EXPECT_EQ(b[0].dataX().imag(), -2.0f);
    EXPECT_EQ(b[1].dataX().real(), -3.0f);
    EXPECT_EQ(b[1].dataX().imag(), -4.0f);
    EXPECT_EQ(b[2].dataX().real(), -5.0f);
    EXPECT_EQ(b[2].dataX().imag(), -6.0f);
}
*/