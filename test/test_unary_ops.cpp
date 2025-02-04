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

#define TEST_VALUE_NEG_OP(TYPE, NAME)                 \
    TEST(ValueTest, Neg##NAME##OP) {                  \
        Value<TYPE> a = 2.0;                          \
                                                      \
        Value<TYPE> b = -a;                           \
        b.backward();                                 \
                                                      \
        EXPECT_EQ(b.dataX(), TYPE(-2.0));             \
        EXPECT_EQ(b.gradX(), TYPE(1.0));              \
        a.zero_grad();                                \
    }

TEST_VALUE_NEG_OP(float, Float)
TEST_VALUE_NEG_OP(double, Double)
TEST_VALUE_NEG_OP(int, Int)


#define TEST_VALUE_NEG(TYPE, NAME)                    \
    TEST(ValueTest, Neg##NAME) {                      \
        Value<TYPE> a = 2.0;                          \
                                                      \
        Value<TYPE> b = ptMgrad::neg(a);              \
        b.backward();                                 \
                                                      \
        EXPECT_EQ(b.dataX(), TYPE(-2.0));             \
        EXPECT_EQ(b.gradX(), TYPE(1.0));              \
        a.zero_grad();                                \
    }

TEST_VALUE_NEG(float, Float)
TEST_VALUE_NEG(double, Double)
TEST_VALUE_NEG(int, Int)


#define TEST_VALUE_NEG_SCALAR(TYPE, NAME)             \
    TEST(ValueTest, Neg##NAME##Scalar) {              \
        TYPE a = 2.0;                                 \
                                                      \
        Value<TYPE> b = ptMgrad::neg(a);              \
                                                      \
        EXPECT_EQ(b.dataX(), TYPE(-2.0));             \
    }

TEST_VALUE_NEG_SCALAR(float, Float)
TEST_VALUE_NEG_SCALAR(double, Double)
TEST_VALUE_NEG_SCALAR(int, Int)


#define TEST_VALUE_NEG_VECTOR(TYPE, NAME)                  \
    TEST(ValueTest, Neg##NAME##Vector) {                   \
        std::vector<Value<TYPE>> a = {2.0, 3.0, 4.0};      \
                                                           \
        std::vector<Value<TYPE>> b = ptMgrad::neg(a);      \
                                                           \
        EXPECT_EQ(b[0].dataX(), TYPE(-2.0));               \
        EXPECT_EQ(b[1].dataX(), TYPE(-3.0));               \
        EXPECT_EQ(b[2].dataX(), TYPE(-4.0));               \
    }

TEST_VALUE_NEG_VECTOR(float, Float)
TEST_VALUE_NEG_VECTOR(double, Double)
TEST_VALUE_NEG_VECTOR(int, Int)


#define TEST_VALUE_NEG_MATRIX(TYPE, NAME)                               \
    TEST(ValueTest, Neg##NAME##Matrix) {                                \
        std::vector<std::vector<Value<TYPE>>> a = {                     \
            {2.0, 3.0, 4.0},                                            \
            {0.0, -6.0, 7.0}                                            \
        };                                                              \
                                                                        \
        std::vector<std::vector<Value<TYPE>>> b = ptMgrad::neg(a);      \
                                                                        \
        EXPECT_EQ(b[0][0].dataX(), TYPE(-2.0));                         \
        EXPECT_EQ(b[0][1].dataX(), TYPE(-3.0));                         \
        EXPECT_EQ(b[0][2].dataX(), TYPE(-4.0));                         \
        EXPECT_EQ(b[1][0].dataX(), TYPE(0.0));                          \
        EXPECT_EQ(b[1][1].dataX(), TYPE(6.0));                          \
        EXPECT_EQ(b[1][2].dataX(), TYPE(-7.0));                         \
    }

TEST_VALUE_NEG_MATRIX(float, Float)
TEST_VALUE_NEG_MATRIX(double, Double)
TEST_VALUE_NEG_MATRIX(int, Int)


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

TEST(ValueTest, ComplexNegScalar) {
    complex<float> a(1.0f, 2.0f);

    Value<complex<float>> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX().real(), -1.0f);
    EXPECT_EQ(b.dataX().imag(), -2.0f);
}

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
