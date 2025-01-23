#include <iostream>

#include <gtest/gtest.h>

#include "../src/engine.h"


using namespace ptMgrad;


// add

TEST(ValueTest, ComplexAdd) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::add(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), 4.0f);
    EXPECT_EQ(c.dataX().imag(), 6.0f);
}

TEST(ValueTest, ComplexAddScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    Value<complex<float>> c = ptMgrad::add(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), 4.0f);
    EXPECT_EQ(c.dataX().imag(), 6.0f);
}

TEST(ValueTest, ComplexAddScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::add(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), 4.0f);
    EXPECT_EQ(c.dataX().imag(), 6.0f);
}

TEST(ValueTest, ComplexAddScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    Value<complex<float>> c = ptMgrad::add(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), 4.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}

TEST(ValueTest, ComplexAddScalar4) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::add(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), 4.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}


// sub

TEST(ValueTest, ComplexSub) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::sub(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), -2.0f);
    EXPECT_EQ(c.dataX().imag(), -2.0f);
}

TEST(ValueTest, ComplexSubScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    Value<complex<float>> c = ptMgrad::sub(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), -2.0f);
    EXPECT_EQ(c.dataX().imag(), -2.0f);
}

TEST(ValueTest, ComplexSubScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::sub(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), -2.0f);
    EXPECT_EQ(c.dataX().imag(), -2.0f);
}

TEST(ValueTest, ComplexSubScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    Value<complex<float>> c = ptMgrad::sub(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), -2.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}

TEST(ValueTest, ComplexSubScalar4) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::sub(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), -2.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}


// rsub

TEST(ValueTest, ComplexRsub) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::rsub(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), 2.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}

TEST(ValueTest, ComplexRsubScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    Value<complex<float>> c = ptMgrad::rsub(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), 2.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}

TEST(ValueTest, ComplexRsubScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::rsub(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), 2.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}

TEST(ValueTest, ComplexRsubScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    Value<complex<float>> c = ptMgrad::rsub(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), 2.0f);
    EXPECT_EQ(c.dataX().imag(), -2.0f);
}

TEST(ValueTest, ComplexRsubScalar4) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::rsub(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), 2.0f);
    EXPECT_EQ(c.dataX().imag(), -2.0f);
}


// mul
/*
TEST(ValueTest, ComplexMul) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::mul(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), -5.0f);
    EXPECT_EQ(c.dataX().imag(), 10.0f);
}
*/


// div
/*
TEST(ValueTest, ComplexDiv) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::div(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), 0.3334f);
    EXPECT_EQ(c.dataX().imag(), 0.5f);
}
*/


// rdiv
/*
TEST(ValueTest, ComplexRdiv) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::rdiv(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), 3.0f);
    EXPECT_EQ(c.dataX().imag(), 2.0f);
}
*/


// pow

TEST(ValueTest, ComplexPow) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::pow(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), 1.0f);
    EXPECT_EQ(c.dataX().imag(), 16.0f);
}

TEST(ValueTest, ComplexPowScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    Value<complex<float>> c = ptMgrad::pow(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), 1.0f);
    EXPECT_EQ(c.dataX().imag(), 16.0f);
}

TEST(ValueTest, ComplexPowScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::pow(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), 1.0f);
    EXPECT_EQ(c.dataX().imag(), 16.0f);
}

TEST(ValueTest, ComplexPowScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    Value<complex<float>> c = ptMgrad::pow(a.dataX(), b);

    EXPECT_EQ(c.dataX().real(), 1.0f);
    EXPECT_EQ(c.dataX().imag(), 8.0f);
}

TEST(ValueTest, ComplexPowScalar4) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::pow(a, b.dataX());

    EXPECT_EQ(c.dataX().real(), 1.0f);
    EXPECT_EQ(c.dataX().imag(), 1.0f);
}


TEST(ValueTest, ComplexPowScalar5) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::pow(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), 1.0f);
    EXPECT_EQ(c.dataX().imag(), 1.0f);   
}