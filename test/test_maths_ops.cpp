#include <iostream>

#include <gtest/gtest.h>


#include "../src/engine.h"


using namespace ptMgrad;

/*
 * Test for mathematical operations:
 *     - addition, substraction, multiplication, division, pow
 * 
 * Type Conversions:
 *     - float x float = float
 *     - double x double = double
 *     - float x double = double
 *     - double x float = double
 *     - float x scalar = float
 *     - double x scalar = double
 *     - scalar x float = float
 *     - scalar x double = double
 *     - scalar x scalar = float (if return type is float)
 *     - scalar x scalar = double (if return type is double)
 */


// addition

// for float x float operations
TEST(ValueTest, FloatAdd) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0f);
}

// for double x double operations
TEST(ValueTest, DoubleAdd) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0);
}

// for float x scalar operations
TEST(ValueTest, FloatAddScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0f);
}

// for double x scalar operations
TEST(ValueTest, DoubleAddScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0);
}

/*
// for scalar x float operations
TEST(ValueTest, ScalarAddFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0f);
}

// for scalar x double operations
TEST(ValueTest, ScalarAddDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0);
}
*/

// for scalar x scalar = float operations
TEST(ValueTest, ScalarAddScalar) {
    float a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0f);
}

// for scalar x scalar = double operations
TEST(ValueTest, ScalarAddScalarDouble) {
    double a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::add(a, b);

    EXPECT_EQ(c.dataX(), 5.0);
}


// substraction

// for float x float operations
TEST(ValueTest, FloatSub) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0f);
}

// for double x double operations
TEST(ValueTest, DoubleSub) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0);
}

// for float x scalar operations
TEST(ValueTest, FloatSubScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0f);
}

// double x scalar operations
TEST(ValueTest, DoubleSubScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0);
}

/*
// for scalar x float = float operations
TEST(ValueTest, ScalarSubFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0f);
}

// for scalar x double = double operations
TEST(ValueTest, ScalarSubDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0);
}
*/

// for scalar x scalar = float operations
TEST(ValueTest, ScalarSubScalar) {
    float a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0f);
}

// for scalar x scalar = double operations
TEST(ValueTest, ScalarSubScalarDouble) {
    double a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c.dataX(), -1.0);
}


// rsub

// for float x float operations
TEST(ValueTest, FloatRsub) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0f);
}

// for double x double operations
TEST(ValueTest, DoubleRsub) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0);
}


// for float x scalar operations
TEST(ValueTest, FloatRsubScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0f);
}

// for double x scalar operations
TEST(ValueTest, DoubleRsubScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0);
}

/*
// for scalar x float = float operations
TEST(ValueTest, ScalarRsubFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0f);
}

// for scalar x double = double operations
TEST(ValueTest, ScalarRsubDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0);
}
*/

// for scalar x scalar = float operations
TEST(ValueTest, ScalarRsubScalar) {
    float a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0f);
}

// for scalar x scalar = double operations
TEST(ValueTest, ScalarRsubScalarDouble) {
    double a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c.dataX(), 1.0);
}


// multiplication

// for float x float operations
TEST(ValueTest, FloatMul) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0f);
}

// for double x double operations
TEST(ValueTest, DoubleMul) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0);
}

// for float x scalar operations
TEST(ValueTest, FloatMulScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0f);
}

// for double x scalar operations
TEST(ValueTest, DoubleMulScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0);
}

/*
// for scalar x float = float operations
TEST(ValueTest, ScalarMulFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0f);
}

// for scalar x double = double operations
TEST(ValueTest, ScalarMulDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0);
}
*/

// for scalar x scalar = float operations
TEST(ValueTest, ScalarMulScalar) {
    float a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0f);
}

// for scalar x scalar = double operations
TEST(ValueTest, ScalarMulScalarDouble) {
    double a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c.dataX(), 6.0);
}


// division

// for float x float operations
TEST(ValueTest, FloatDiv) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0f / 3.0f, 0.001);
}

// for double x double operations
TEST(ValueTest, DoubleDiv) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0 / 3.0, 0.001);
}

// for float x scalar = float operations
TEST(ValueTest, FloatDivScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0f / 3.0f, 0.001);
}

// for double x scalar = double operations
TEST(ValueTest, DoubleDivScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0 / 3.0, 0.001);
}

/*
// for scalar x float = float operations
TEST(ValueTest, ScalarDivFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0f / 3.0f, 0.001);
}

// for scalar x double = double operations
TEST(ValueTest, ScalarDivDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0 / 3.0, 0.001);
}
*/

// for scalar x scalar = float operations
TEST(ValueTest, ScalarDivScalar) {
    float a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0f / 3.0f, 0.001);
}

// for scalar x scalar = double operations
TEST(ValueTest, ScalarDivScalarDouble) {
    double a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::div(a, b);

    EXPECT_NEAR(c.dataX(), 2.0 / 3.0, 0.001);
}


// rdiv

// for float x float operations
TEST(ValueTest, FloatRdiv) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5f);
}

// for double x double operations
TEST(ValueTest, DoubleRdiv) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5);
}


// for float x scalar operations
TEST(ValueTest, FloatRdivScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5f);
}

// for double x scalar operations
TEST(ValueTest, DoubleRdivScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5);
}

/*
// for scalar x float = float operations
TEST(ValueTest, ScalarRdivFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5f);
}

// for scalar x double = double operations
TEST(ValueTest, ScalarRdivDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5);
}
*/

// for scalar x scalar = float operations
TEST(ValueTest, ScalarRdivScalar) {
    float a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5f);
}

// for scalar x scalar = double operations
TEST(ValueTest, ScalarRdivScalarDouble) {
    double a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c.dataX(), 1.5);
}


// Pow

// for float x float operations
TEST(ValueTest, FloatPow) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0f);
}

// for double x double operations
TEST(ValueTest, DoublePow) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0);
}

// for float x scalar operations
TEST(ValueTest, FloatPowScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0f);
}

// for double x scalar operations
TEST(ValueTest, DoublePowScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0);
}

/*
// for scalar x float operations
TEST(ValueTest, ScalarPowFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    Value<float> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0f);
}

// for scalar x double operations
TEST(ValueTest, ScalarPowDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    Value<double> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0);
}
*/

// for scalar x scalar = float operations
TEST(ValueTest, ScalarPowScalarFloat) {
    float a = 2.0f;
    float b = 3.0f;

    Value<float> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0f);
}

// for scalar x scalar = double operations
TEST(ValueTest, ScalarPowScalarDouble) {
    double a = 2.0;
    double b = 3.0;

    Value<double> c = ptMgrad::pow(a, b);

    EXPECT_EQ(c.dataX(), 8.0);
}
