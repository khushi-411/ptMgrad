#include <iostream>

#include <gtest/gtest.h>


#include "../src/engine.h"


using namespace ptMgrad;


/*
 * Test for comparison operations:
 *     - lt, gt
 * 
 * Type Conversions:
 *     - type1 x type2 = bool
 */


// lt

TEST(ValueTest, FloatLt) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    bool c = ptMgrad::lt(a, b);

    EXPECT_EQ(c, true);
}

TEST(ValueTest, DoubleLt) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    bool c = ptMgrad::lt(a, b);

    EXPECT_EQ(c, true);
}
/*
TEST(ValueTest, ScalarLtFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    bool c = ptMgrad::lt(a, b);

    EXPECT_EQ(c, true);
}
*/

// gt

TEST(ValueTest, FloatGt) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    bool c = ptMgrad::gt(a, b);

    EXPECT_EQ(c, false);
}

TEST(ValueTest, DoubleGt) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    bool c = ptMgrad::gt(a, b);

    EXPECT_EQ(c, false);
}
/*
TEST(ValueTest, ScalarGtFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    bool c = ptMgrad::gt(a, b);

    EXPECT_EQ(c, false);
}
*/