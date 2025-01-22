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

TEST(ValueTest, FloatNeg) {
    Value<float> a = 2.0f;

    Value<float> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX(), -2.0f);
}

TEST(ValueTest, DoubleNeg) {
    Value<double> a = 2.0;

    Value<double> b = ptMgrad::neg(a);

    EXPECT_EQ(b.dataX(), -2.0);
}

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