#include <iostream>

#include <gtest/gtest.h>


#include "../src/engine.h"


using namespace ptMgrad;


/*
 * Test for activation functions:
 *     - relu
 */


// relu

TEST(ValueTest, FloatRelu) {
    Value<float> a = 2.0f;

    Value<float> b = ptMgrad::relu(a);

    EXPECT_EQ(b.dataX(), 2.0f);
}


TEST(ValueTest, DoubleRelu) {
    Value<double> a = 2.0;

    Value<double> b = ptMgrad::relu(a);

    EXPECT_EQ(b.dataX(), 2.0);
}


TEST(ValueTest, FloatReluArray) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};

    std::vector<Value<float>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0].dataX(), 2.0f);
    EXPECT_EQ(b[1].dataX(), 3.0f);
    EXPECT_EQ(b[2].dataX(), 4.0f);
}


TEST(ValueTest, DoubleReluArray) {
    std::vector<Value<double>> a = {2.0, 3.0, 4.0};

    std::vector<Value<double>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0].dataX(), 2.0);
    EXPECT_EQ(b[1].dataX(), 3.0);
    EXPECT_EQ(b[2].dataX(), 4.0);
}


TEST(ValueTest, FloatReluArrayNegative) {
    std::vector<Value<float>> a = {-2.0f, -3.0f, -4.0f};

    std::vector<Value<float>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0].dataX(), 0.0f);
    EXPECT_EQ(b[1].dataX(), 0.0f);
    EXPECT_EQ(b[2].dataX(), 0.0f);
}


TEST(ValueTest, DoubleReluArrayNegative) {
    std::vector<Value<double>> a = {-2.0, -3.0, -4.0};

    std::vector<Value<double>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0].dataX(), 0.0);
    EXPECT_EQ(b[1].dataX(), 0.0);
    EXPECT_EQ(b[2].dataX(), 0.0);
}


TEST(ValueTest, FloatRelu2DArray) {
    std::vector<std::vector<Value<float>>> a = {{2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f}};

    std::vector<std::vector<Value<float>>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0][0].dataX(), 2.0f);
    EXPECT_EQ(b[0][1].dataX(), 3.0f);
    EXPECT_EQ(b[0][2].dataX(), 4.0f);
    EXPECT_EQ(b[1][0].dataX(), 5.0f);
    EXPECT_EQ(b[1][1].dataX(), 6.0f);
    EXPECT_EQ(b[1][2].dataX(), 7.0f);
}


TEST(ValueTest, DoubleRelu2DArray) {
    std::vector<std::vector<Value<double>>> a = {{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};

    std::vector<std::vector<Value<double>>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0][0].dataX(), 2.0);
    EXPECT_EQ(b[0][1].dataX(), 3.0);
    EXPECT_EQ(b[0][2].dataX(), 4.0);
    EXPECT_EQ(b[1][0].dataX(), 5.0);
    EXPECT_EQ(b[1][1].dataX(), 6.0);
    EXPECT_EQ(b[1][2].dataX(), 7.0);
}


TEST(ValueTest, FloatValueRelu1DArray) {
    std::vector<Value<float>> a = {Value<float>(2.0f), Value<float>(-3.0f), Value<float>(4.0f)};

    std::vector<Value<float>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0].dataX(), 2.0f);
    EXPECT_EQ(b[1].dataX(), 0.0f);
    EXPECT_EQ(b[2].dataX(), 4.0f);
}


TEST(ValueTest, DoubleValueRelu1DArray) {
    std::vector<Value<double>> a = {Value<double>(2.0), Value<double>(-3.0), Value<double>(4.0)};

    std::vector<Value<double>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0].dataX(), 2.0);
    EXPECT_EQ(b[1].dataX(), 0.0);
    EXPECT_EQ(b[2].dataX(), 4.0);
}


TEST(ValueTest, FloatValueRelu2DArray) {
    std::vector<std::vector<Value<float>>> a = {
        {Value<float>(2.0f), Value<float>(-3.0f), Value<float>(4.0f)},
        {Value<float>(5.0f), Value<float>(6.0f), Value<float>(-7.0f)}
    };

    std::vector<std::vector<Value<float>>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0][0].dataX(), 2.0f);
    EXPECT_EQ(b[0][1].dataX(), 0.0f);
    EXPECT_EQ(b[0][2].dataX(), 4.0f);
    EXPECT_EQ(b[1][0].dataX(), 5.0f);
    EXPECT_EQ(b[1][1].dataX(), 6.0f);
    EXPECT_EQ(b[1][2].dataX(), 0.0f);
}


TEST(ValueTest, DoubleValueRelu2DArray) {
    std::vector<std::vector<Value<double>>> a = {
        {Value<double>(2.0), Value<double>(-3.0), Value<double>(4.0)},
        {Value<double>(5.0), Value<double>(6.0), Value<double>(-7.0)}
    };

    std::vector<std::vector<Value<double>>> b = ptMgrad::relu(a);

    EXPECT_EQ(b[0][0].dataX(), 2.0);
    EXPECT_EQ(b[0][1].dataX(), 0.0);
    EXPECT_EQ(b[0][2].dataX(), 4.0);
    EXPECT_EQ(b[1][0].dataX(), 5.0);
    EXPECT_EQ(b[1][1].dataX(), 6.0);
    EXPECT_EQ(b[1][2].dataX(), 0.0);
}