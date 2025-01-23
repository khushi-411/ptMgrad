#include <iostream>

#include <gtest/gtest.h>

#include "../src/engine.h"


using namespace ptMgrad;

// add

TEST(ValueTest, ComplexAddOp) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.dataX().real(), 3.0f);
    EXPECT_EQ(b.dataX().imag(), 4.0f);

    EXPECT_EQ(a + b, complex<float>(4.0f, 6.0f));
}

TEST(ValueTest, ComplexAddOpScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.real(), 3.0f);
    EXPECT_EQ(b.imag(), 4.0f);

    EXPECT_EQ(a + b, complex<float>(4.0f, 6.0f));
}

TEST(ValueTest, ComplexAddOpScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    EXPECT_EQ(a.real(), 1.0f);
    EXPECT_EQ(a.imag(), 2.0f);
    EXPECT_EQ(b.dataX().real(), 3.0f);
    EXPECT_EQ(b.dataX().imag(), 4.0f);

    EXPECT_EQ(a + b, complex<float>(4.0f, 6.0f));
}

/*
TEST(ValueTest, ComplexAddOpScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);

    EXPECT_EQ(a + b, complex<float>(4.0f, 5.0f));
}
*/


// sub

TEST(ValueTest, ComplexSubOp) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.dataX().real(), 3.0f);
    EXPECT_EQ(b.dataX().imag(), 4.0f);

    EXPECT_EQ(a - b, complex<float>(-2.0f, -2.0f));
}

TEST(ValueTest, ComplexSubOpScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.real(), 3.0f);
    EXPECT_EQ(b.imag(), 4.0f);

    EXPECT_EQ(a - b, complex<float>(-2.0f, -2.0f));
}

TEST(ValueTest, ComplexSubOpScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    EXPECT_EQ(a.real(), 1.0f);
    EXPECT_EQ(a.imag(), 2.0f);
    EXPECT_EQ(b.dataX().real(), 3.0f);
    EXPECT_EQ(b.dataX().imag(), 4.0f);

    EXPECT_EQ(a - b, complex<float>(-2.0f, -2.0f));
}

/*
TEST(ValueTest, ComplexSubOpScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);

    EXPECT_EQ(a - b, complex<float>(-2.0f, -2.0f));
}
*/


// mul
/*
TEST(ValueTest, ComplexMulOp) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.dataX().real(), 3.0f);
    EXPECT_EQ(b.dataX().imag(), 4.0f);

    EXPECT_EQ(a * b, complex<float>(-5.0f, 10.0f));
}

TEST(ValueTest, ComplexMulOpScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.real(), 3.0f);
    EXPECT_EQ(b.imag(), 4.0f);

    EXPECT_EQ(a * b, complex<float>(-5.0f, 10.0f));
}
*/


// div
/*
TEST(ValueTest, ComplexDivOp) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    EXPECT_EQ(a.dataX().real(), 1.0f);
    EXPECT_EQ(a.dataX().imag(), 2.0f);
    EXPECT_EQ(b.dataX().real(), 3.0f);
    EXPECT_EQ(b.dataX().imag(), 4.0f);

    EXPECT_EQ(a / b, complex<float>(0.44f, 0.08f));
}
*/