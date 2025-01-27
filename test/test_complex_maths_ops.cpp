#include <iostream>

#include <gtest/gtest.h>

#include "../src/engine.h"


using namespace ptMgrad;


// add

TEST(ValueTest, ComplexAdd) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::add(a, b);

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

TEST(ValueTest, ComplexAddVector) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::add(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(6.0f, 8.0f),
        complex<float>(10.0f, 12.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

TEST(ValueTest, ComplexAddVectorScalar) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    complex<float> b(5.0f, 6.0f);

    std::vector<Value<complex<float>>> c = ptMgrad::add(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(6.0f, 8.0f),
        complex<float>(8.0f, 10.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

/*
TEST(ValueTest, ComplexAddVectorScalar2) {
    complex<float> a(1.0f, 2.0f);
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::add(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(6.0f, 8.0f),
        complex<float>(8.0f, 10.0f)
    };

    EXPECT_EQ(c, result);
}
*/
/*
TEST(ValueTest, ComplexAddVectorScalar3) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    float b = 5.0f;

    std::vector<Value<complex<float>>> c = ptMgrad::add(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(6.0f, 2.0f),
        complex<float>(8.0f, 4.0f)
    };

    EXPECT_EQ(c, result);
}
*/

TEST(ValueTest, ComplexMatrixAdd) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };

    std::vector<std::vector<Value<complex<float>>>> b = {
        {complex<float>(9.0f, 10.0f), complex<float>(11.0f, 12.0f)},
        {complex<float>(13.0f, 14.0f), complex<float>(15.0f, 16.0f)}
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::add(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(10.0f, 12.0f), complex<float>(14.0f, 16.0f)},
        {complex<float>(18.0f, 20.0f), complex<float>(22.0f, 24.0f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
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

TEST(ValueTest, ComplexSubVector) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::sub(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(-4.0f, -4.0f),
        complex<float>(-4.0f, -4.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

TEST(ValueTest, ComplexSubVectorScalar) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    complex<float> b(5.0f, 6.0f);

    std::vector<Value<complex<float>>> c = ptMgrad::sub(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(-4.0f, -4.0f),
        complex<float>(-2.0f, -2.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

TEST(ValueTest, ComplexMatrixSub) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };
    std::vector<std::vector<Value<complex<float>>>> b = {
        {complex<float>(9.0f, 10.0f), complex<float>(11.0f, 12.0f)},
        {complex<float>(13.0f, 14.0f), complex<float>(15.0f, 16.0f)}
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::sub(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(-8.0f, -8.0f), complex<float>(-8.0f, -8.0f)},
        {complex<float>(-8.0f, -8.0f), complex<float>(-8.0f, -8.0f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}

TEST(ValueTest, ComplexMatrixScalarSub) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };
    complex<float> b(9.0f, 10.0f);

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::sub(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(-8.0f, -8.0f), complex<float>(-6.0f, -6.0f)},
        {complex<float>(-4.0f, -4.0f), complex<float>(-2.0f, -2.0f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}

/*
TEST(ValueTest, ComplexMatrixVectorSub) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };

    std::vector<Value<complex<float>>> b = {
        complex<float>(9.0f, 10.0f),
        complex<float>(11.0f, 12.0f)
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::sub(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(-8.0f, -8.0f), complex<float>(-8.0f, -8.0f)},
        {complex<float>(-4.0f, -4.0f), complex<float>(-4.0f, -4.0f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}
*/


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

TEST(ValueTest, ComplexRsubVector) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::rsub(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(4.0f, 4.0f),
        complex<float>(4.0f, 4.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

TEST(ValueTest, ComplexRsubMatrix) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };
    std::vector<std::vector<Value<complex<float>>>> b = {
        {complex<float>(16.0f, 17.0f), complex<float>(18.0f, 19.0f)},
        {complex<float>(20.0f, 21.0f), complex<float>(22.0f, 23.0f)}
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::rsub(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(15.0f, 15.0f), complex<float>(15.0f, 15.0f)},
        {complex<float>(15.0f, 15.0f), complex<float>(15.0f, 15.0f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}

/*
TEST(ValueTest, ComplexRsubMatrixVector) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };
    std::vector<complex<float>> b = {
        complex<float>(-1.0f, -2.0f),
        complex<float>(-3.0f, -4.0f)
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::rsub(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(-2.0f, -4.0f), complex<float>(-6.0f, -8.0f)},
        {complex<float>(-6.0f, -8.0f), complex<float>(-10.0f, -10.0f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}
*/


// mul

TEST(ValueTest, ComplexMul) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::mul(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX().real(), -5.0f);
    EXPECT_EQ(c.dataX().imag(), 10.0f);
}

TEST(ValueTest, ComplexMulScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    Value<complex<float>> c = ptMgrad::mul(a.dataX(), b);

    EXPECT_EQ(c.dataX(), complex<float>(-5.0f, 10.0f));
}

TEST(ValueTest, ComplexMulScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::mul(a, b.dataX());

    EXPECT_EQ(c.dataX(), complex<float>(-5.0f, 10.0f));
}

TEST(ValueTest, ComplexMulScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    Value<complex<float>> c = ptMgrad::mul(a.dataX(), b);

    EXPECT_EQ(c.dataX(), complex<float>(3.0f, 6.0f));
}

TEST(ValueTest, ComplexMulScalar4) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::mul(a, b.dataX());

    EXPECT_EQ(c.dataX(), complex<float>(3.0f, 6.0f));
}

TEST(ValueTest, ComplexMulVector) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };

    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::mul(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(7.0f, 10.0f),
        complex<float>(21.0f, 26.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
}

TEST(ValueTest, ComplexMulVectorScalar) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    complex<float> b(5.0f, 6.0f);

    std::vector<Value<complex<float>>> c = ptMgrad::mul(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(7.0f, 10.0f),
        complex<float>(21.0f, 26.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
}

/*
TEST(ValueTest, ComplexMulVectorScalar2) {
    complex<float> a(1.0f, 2.0f);
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::mul(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(7.0f, 10.0f),
        complex<float>(21.0f, 26.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
}
*/

TEST(ValueTest, ComplexMulMatrix) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };
    std::vector<std::vector<Value<complex<float>>>> b = {
        {complex<float>(9.0f, 10.0f), complex<float>(11.0f, 12.0f)},
        {complex<float>(13.0f, 14.0f), complex<float>(15.0f, 16.0f)}
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::mul(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(-11.0f, 20.0f), complex<float>(-29.0f, 44.0f)},
        {complex<float>(-53.0f, 80.0f), complex<float>(-85.0f, 128.0f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}


// div

TEST(ValueTest, ComplexDiv) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::div(a.dataX(), b.dataX());

    EXPECT_EQ(c, Value<complex<float>>(0.44f, 0.08f));
}

TEST(ValueTest, ComplexDivScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    Value<complex<float>> c = ptMgrad::div(a.dataX(), b);

    EXPECT_EQ(c, Value<complex<float>>(0.44f, 0.08f));
}

TEST(ValueTest, ComplexDivScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::div(a, b.dataX());

    EXPECT_EQ(c, Value<complex<float>>(0.44f, 0.08f));
}

TEST(ValueTest, ComplexDivScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    Value<complex<float>> c = ptMgrad::div(a.dataX(), b);

    EXPECT_EQ(c, Value<complex<float>>(0.33f, 0.67f));
}

TEST(ValueTest, ComplexDivScalar4) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::div(a, b.dataX());

    EXPECT_EQ(c, Value<complex<float>>(0.33f, 0.67f));
}

TEST(ValueTest, ComplexDivScalar5) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<float> b(3.0f);

    Value<complex<float>> c = ptMgrad::div(a.dataX(), b.dataX());

    EXPECT_EQ(c, Value<complex<float>>(0.33f, 0.67f));
}

TEST(ValueTest, ComplexDivVector) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::div(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(0.44f, 0.08f),
        complex<float>(0.44f, 0.08f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

TEST(ValueTest, ComplexDivVectorScalar) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    complex<float> b(5.0f, 6.0f);

    std::vector<Value<complex<float>>> c = ptMgrad::div(a, b);
    std::vector<Value<complex<float>>> result = {
        complex<float>(0.44f, 0.08f),
        complex<float>(0.44f, 0.08f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

TEST(ValueTest, ComplexDivMatrix) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };
    std::vector<std::vector<Value<complex<float>>>> b = {
        {complex<float>(9.0f, 10.0f), complex<float>(11.0f, 12.0f)},
        {complex<float>(13.0f, 14.0f), complex<float>(15.0f, 16.0f)}
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::div(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(0.44f, 0.08f), complex<float>(0.44f, 0.08f)},
        {complex<float>(0.44f, 0.08f), complex<float>(0.44f, 0.08f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}


// rdiv

TEST(ValueTest, ComplexRdiv) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::rdiv(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX(), complex<float>(0.44f, -0.08f));
}

TEST(ValueTest, ComplexRdivScalar) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    complex<float> b(3.0f, 4.0f);

    Value<complex<float>> c = ptMgrad::rdiv(a.dataX(), b);

    EXPECT_EQ(c.dataX(), complex<float>(0.44f, -0.08f));
}

TEST(ValueTest, ComplexRdivScalar2) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(complex<float>(3.0f, 4.0f));

    Value<complex<float>> c = ptMgrad::rdiv(a, b.dataX());

    EXPECT_EQ(c.dataX(), complex<float>(0.44f, -0.08f));
}

TEST(ValueTest, ComplexRdivScalar3) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    float b = 3.0f;

    Value<complex<float>> c = ptMgrad::rdiv(a.dataX(), b);

    EXPECT_EQ(c.dataX(), complex<float>(0.33f, -0.67f));
}

TEST(ValueTest, ComplexRdivScalar4) {
    complex<float> a(1.0f, 2.0f);
    Value<complex<float>> b(3.0f);

    Value<complex<float>> c = ptMgrad::rdiv(a, b.dataX());

    EXPECT_EQ(c.dataX(), complex<float>(0.33f, -0.67f));
}

TEST(ValueTest, ComplexRdivScalar5) {
    Value<complex<float>> a(complex<float>(1.0f, 2.0f));
    Value<float> b(3.0f);

    Value<complex<float>> c = ptMgrad::rdiv(a.dataX(), b.dataX());

    EXPECT_EQ(c.dataX(), complex<float>(0.33f, -0.67f));
}

TEST(ValueTest, ComplexRdivVector) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::rdiv(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(0.44f, -0.08f),
        complex<float>(0.44f, -0.08f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}

TEST(ValueTest, ComplexRdivVectorScalar) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    complex<float> b(5.0f, 6.0f);

    std::vector<Value<complex<float>>> c = ptMgrad::rdiv(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(0.44f, -0.08f),
        complex<float>(0.44f, -0.08f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}
/*
TEST(ValueTest, ComplexRdivMatrix) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };
    std::vector<std::vector<Value<complex<float>>>> b = {
        {complex<float>(9.0f, 10.0f), complex<float>(11.0f, 12.0f)},
        {complex<float>(13.0f, 14.0f), complex<float>(15.0f, 16.0f)}
    };

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::rdiv(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(0.44f, -0.08f), complex<float>(0.44f, -0.08f)},
        {complex<float>(0.44f, -0.08f), complex<float>(0.44f, -0.08f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
}

TEST(ValueTest, ComplexRdivMatrixScalar) {
    std::vector<std::vector<Value<complex<float>>>> a = {
        {complex<float>(1.0f, 2.0f), complex<float>(3.0f, 4.0f)},
        {complex<float>(5.0f, 6.0f), complex<float>(7.0f, 8.0f)}
    };

    complex<float> b(9.0f, 10.0f);

    std::vector<std::vector<Value<complex<float>>>> c = ptMgrad::rdiv(a, b);

    std::vector<std::vector<Value<complex<float>>>> result = {
        {complex<float>(0.44f, -0.08f), complex<float>(0.44f, -0.08f)},
        {complex<float>(0.44f, -0.08f), complex<float>(0.44f, -0.08f)}
    };

    EXPECT_EQ(c[0][0].dataX().real(), result[0][0].dataX().real());
    EXPECT_EQ(c[0][0].dataX().imag(), result[0][0].dataX().imag());

    EXPECT_EQ(c[0][1].dataX().real(), result[0][1].dataX().real());
    EXPECT_EQ(c[0][1].dataX().imag(), result[0][1].dataX().imag());

    EXPECT_EQ(c[1][0].dataX().real(), result[1][0].dataX().real());
    EXPECT_EQ(c[1][0].dataX().imag(), result[1][0].dataX().imag());

    EXPECT_EQ(c[1][1].dataX().real(), result[1][1].dataX().real());
    EXPECT_EQ(c[1][1].dataX().imag(), result[1][1].dataX().imag());
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

/*
TEST(ValueTest, ComplexPowVector) { 
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    std::vector<Value<complex<float>>> b = {
        complex<float>(5.0f, 6.0f),
        complex<float>(7.0f, 8.0f)
    };

    std::vector<Value<complex<float>>> c = ptMgrad::pow(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(1.0f, 16.0f),
        complex<float>(1.0f, 16.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}


TEST(ValueTest, ComplexPowVectorScalar) {
    std::vector<Value<complex<float>>> a = {
        complex<float>(1.0f, 2.0f),
        complex<float>(3.0f, 4.0f)
    };
    complex<float> b(5.0f, 6.0f);

    std::vector<Value<complex<float>>> c = ptMgrad::pow(a, b);

    std::vector<Value<complex<float>>> result = {
        complex<float>(1.0f, 16.0f),
        complex<float>(1.0f, 16.0f)
    };

    EXPECT_EQ(c[0].dataX().real(), result[0].dataX().real());
    EXPECT_EQ(c[0].dataX().imag(), result[0].dataX().imag());
    EXPECT_EQ(c[1].dataX().real(), result[1].dataX().real());
    EXPECT_EQ(c[1].dataX().imag(), result[1].dataX().imag());
}
*/
