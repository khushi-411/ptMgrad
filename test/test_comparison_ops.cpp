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

TEST(ValueTest, ScalarLtScalar) {
    float a = 2.0f;
    float b = 3.0f;

    bool c = ptMgrad::lt(a, b);

    EXPECT_EQ(c, true);
}

TEST(ValueTest, VectorLtVector) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    std::vector<Value<float>> b = {5.0f, -6.0f, 7.0f};

    std::vector<bool> c = ptMgrad::lt(a, b);

    EXPECT_EQ(c[0], true);
    EXPECT_EQ(c[1], false);
    EXPECT_EQ(c[2], true);
}

TEST(ValueTest, VectorLtScalar) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    float b = 5.0f;

    std::vector<bool> c = ptMgrad::lt(a, b);

    EXPECT_EQ(c[0], true);
    EXPECT_EQ(c[1], true);
    EXPECT_EQ(c[2], true);
}

TEST(ValueTest, MatrixLtMatrix) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    std::vector<std::vector<Value<float>>> b = {
        {5.0f, -6.0f, 7.0f},
        {2.0f, 3.0f, 4.0f}
    };

    std::vector<std::vector<bool>> c = ptMgrad::lt(a, b);

    EXPECT_EQ(c[0][0], true);
    EXPECT_EQ(c[0][1], false);
    EXPECT_EQ(c[0][2], true);
    EXPECT_EQ(c[1][0], false);
    EXPECT_EQ(c[1][1], true);
    EXPECT_EQ(c[1][2], false);
}

TEST(ValueTest, MatrixLtMatrixVector) {
    std::vector<std::vector<Value<float>>> a = {
        {Value<float>(2.0f), Value<float>(3.0f), Value<float>(4.0f)},
        {Value<float>(5.0f), Value<float>(-6.0f), Value<float>(7.0f)}
    };

    std::vector<std::vector<Value<float>>> b = {
        {Value<float>(5.0f), Value<float>(-6.0f), Value<float>(7.0f)},
        {Value<float>(2.0f), Value<float>(3.0f), Value<float>(4.0f)}
    };

    std::vector<std::vector<bool>> c = ptMgrad::lt(a, b);

    EXPECT_EQ(c[0][0], true);
    EXPECT_EQ(c[0][1], false);
    EXPECT_EQ(c[0][2], true);
    EXPECT_EQ(c[1][0], false);
    EXPECT_EQ(c[1][1], true);
    EXPECT_EQ(c[1][2], false);
}

TEST(ValueTest, MatrixLtMatrixScalar) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    float b = 5.0f;

    std::vector<std::vector<bool>> c = ptMgrad::lt(a, b);

    EXPECT_EQ(c[0][0], true);
    EXPECT_EQ(c[0][1], true);
    EXPECT_EQ(c[0][2], true);
    //EXPECT_EQ(c[1][0], false);
    //EXPECT_EQ(c[1][1], true);
    //EXPECT_EQ(c[1][2], false);
}

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

TEST(ValueTest, ScalarGtScalar) {
    float a = 2.0f;
    float b = 3.0f;

    bool c = ptMgrad::gt(a, b);

    EXPECT_EQ(c, false);
}

TEST(ValueTest, VectorGtVector) {
    std::vector<Value<float>> a = {
        Value<float>(2.0f),
        Value<float>(3.0f),
        Value<float>(4.0f)
    };
    std::vector<Value<float>> b = {
        Value<float>(5.0f),
        Value<float>(-6.0f),
        Value<float>(7.0f)
    };

    std::vector<bool> c = ptMgrad::gt(a, b);

    EXPECT_EQ(c[0], false);
    EXPECT_EQ(c[1], false);  // TODO: check this 
    EXPECT_EQ(c[2], false);
}

TEST(ValueTest, VectorGtVectorScalar) {
    std::vector<Value<float>> a = {
        Value<float>(2.0f),
        Value<float>(3.0f),
        Value<float>(4.0f)
    };
    std::vector<Value<float>> b = {5.0f, -6.0f, 7.0f};

    std::vector<bool> c = ptMgrad::gt(a, b);

    EXPECT_EQ(c[0], false);
    EXPECT_EQ(c[1], false);  // TODO: check this
    EXPECT_EQ(c[2], false);
}

TEST(ValueTest, VectorGtScalar) {
    std::vector<Value<float>> a = {
        Value<float>(2.0f),
        Value<float>(3.0f),
        Value<float>(4.0f)
    };
    float b = 5.0f;

    std::vector<bool> c = ptMgrad::gt(a, b);

    EXPECT_EQ(c[0], false);
    EXPECT_EQ(c[1], false);
    EXPECT_EQ(c[2], false);
}

TEST(ValueTest, VectorGtScalarScalar) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    float b = 5.0f;

    std::vector<bool> c = ptMgrad::gt(a, b);

    EXPECT_EQ(c[0], false);
    EXPECT_EQ(c[1], false);
    EXPECT_EQ(c[2], false);
}

TEST(ValueTest, MatrixGtMatrix) {
    std::vector<std::vector<Value<float>>> a = {
        {Value<float>(2.0f), Value<float>(3.0f), Value<float>(4.0f)},
        {Value<float>(5.0f), Value<float>(-6.0f), Value<float>(7.0f)}
    };
    std::vector<std::vector<Value<float>>> b = {
        {Value<float>(5.0f), Value<float>(-6.0f), Value<float>(7.0f)},
        {Value<float>(2.0f), Value<float>(3.0f), Value<float>(4.0f)}
    };

    std::vector<std::vector<bool>> c = ptMgrad::gt(a, b);

    EXPECT_EQ(c[0][0], false);
    EXPECT_EQ(c[0][1], false);
    EXPECT_EQ(c[0][2], false);
    EXPECT_EQ(c[1][0], false);  // TODO: check this
    EXPECT_EQ(c[1][1], false);
    EXPECT_EQ(c[1][2], false);  // TODO: check this
}

TEST(ValueTest, MatrixGtScalar) {
    std::vector<std::vector<Value<float>>> a = {
        {Value<float>(2.0f), Value<float>(3.0f), Value<float>(4.0f)},
        {Value<float>(5.0f), Value<float>(-6.0f), Value<float>(7.0f)}
    };
    float b = 5.0f;

    std::vector<std::vector<bool>> c = ptMgrad::gt(a, b);

    EXPECT_EQ(c[0][0], false);
    EXPECT_EQ(c[0][1], false);
    EXPECT_EQ(c[0][2], false);
    EXPECT_EQ(c[1][0], false);  // TODO: check this
    EXPECT_EQ(c[1][1], false);
    EXPECT_EQ(c[1][2], false);  // TODO: check this
}