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