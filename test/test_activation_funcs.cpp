#include <iostream>

#include <gtest/gtest.h>


#include "../src/engine.h"


using namespace ptMgrad;


/*
 * Test for activation functions:
 *     - relu
 */


// relu

#define TEST_VALUE_RELU(TYPE, NAME)                    \
    TEST(ValueTest, Relu##NAME) {                      \
        Value<TYPE> a = 2.0;                           \
                                                       \
        Value<TYPE> b = ptMgrad::relu(a);              \
        b.backward();                                  \
                                                       \
        EXPECT_EQ(b.dataX(), TYPE(2.0));               \
        EXPECT_EQ(a.gradX(), TYPE(1.0));               \
        EXPECT_EQ(b.gradX(), TYPE(1.0));               \
        a.zero_grad();                                 \
        b.zero_grad();                                 \
    }

TEST_VALUE_RELU(float, Float)
TEST_VALUE_RELU(double, Double)
TEST_VALUE_RELU(int, Int)


#define TEST_VALUE_RELU_SCALAR(TYPE, NAME)            \
    TEST(ValueTest, Relu##NAME##Scalar) {             \
        TYPE a = 2.0;                                 \
                                                      \
        Value<TYPE> b = ptMgrad::relu(a);             \
                                                      \
        EXPECT_EQ(b.dataX(), TYPE(2.0));              \
    }

TEST_VALUE_RELU_SCALAR(float, Float)
TEST_VALUE_RELU_SCALAR(double, Double)
TEST_VALUE_RELU_SCALAR(int, Int)


#define TEST_VALUE_RELU_ARRAY(TYPE, NAME)                    \
    TEST(ValueTest, Relu##NAME##Array) {                     \
        std::vector<Value<TYPE>> a = {2.0, 3.0, 4.0};        \
                                                             \
        std::vector<Value<TYPE>> b = ptMgrad::relu(a);       \
                                                             \
        EXPECT_EQ(b[0].dataX(), TYPE(2.0));                  \
        EXPECT_EQ(b[1].dataX(), TYPE(3.0));                  \
        EXPECT_EQ(b[2].dataX(), TYPE(4.0));                  \
    }

TEST_VALUE_RELU_ARRAY(float, Float)
TEST_VALUE_RELU_ARRAY(double, Double)
TEST_VALUE_RELU_ARRAY(int, Int)


#define TEST_VALUE_RELU_ARRAY_NEGATIVE(TYPE, NAME)            \
    TEST(ValueTest, Relu##NAME##ArrayNegative) {              \
        std::vector<Value<TYPE>> a = {-2.0, -3.0, -4.0};      \
                                                              \
        std::vector<Value<TYPE>> b = ptMgrad::relu(a);        \
                                                              \
        EXPECT_EQ(b[0].dataX(), TYPE(0.0));                   \
        EXPECT_EQ(b[1].dataX(), TYPE(0.0));                   \
        EXPECT_EQ(b[2].dataX(), TYPE(0.0));                   \
    }

TEST_VALUE_RELU_ARRAY_NEGATIVE(float, Float)
TEST_VALUE_RELU_ARRAY_NEGATIVE(double, Double)
TEST_VALUE_RELU_ARRAY_NEGATIVE(int, Int)


#define TEST_VALUE_RELU_2D_ARRAY(TYPE, NAME)                                                \
    TEST(ValueTest, Relu##NAME##2DArray) {                                                  \
        std::vector<std::vector<Value<TYPE>>> a = {{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};       \
                                                                                            \
        std::vector<std::vector<Value<TYPE>>> b = ptMgrad::relu(a);                         \
                                                                                            \
        EXPECT_EQ(b[0][0].dataX(), TYPE(2.0));                                              \
        EXPECT_EQ(b[0][1].dataX(), TYPE(3.0));                                              \
        EXPECT_EQ(b[0][2].dataX(), TYPE(4.0));                                              \
        EXPECT_EQ(b[1][0].dataX(), TYPE(5.0));                                              \
        EXPECT_EQ(b[1][1].dataX(), TYPE(6.0));                                              \
        EXPECT_EQ(b[1][2].dataX(), TYPE(7.0));                                              \
    }

TEST_VALUE_RELU_2D_ARRAY(float, Float)
TEST_VALUE_RELU_2D_ARRAY(double, Double)
TEST_VALUE_RELU_2D_ARRAY(int, Int)


#define TEST_VALUE_RELU_1D_ARRAY_VALUE(TYPE, NAME)                           \
    TEST(ValueTest, Relu##NAME##1DArrayValue) {                              \
        std::vector<Value<TYPE>> a = {                                       \
            Value<TYPE>(2.0), Value<TYPE>(-3.0), Value<TYPE>(4.0)            \
        };                                                                   \
                                                                             \
        std::vector<Value<TYPE>> b = ptMgrad::relu(a);                       \
                                                                             \
        EXPECT_EQ(b[0].dataX(), TYPE(2.0));                                  \
        EXPECT_EQ(b[1].dataX(), TYPE(0.0));                                  \
        EXPECT_EQ(b[2].dataX(), TYPE(4.0));                                  \
    }

TEST_VALUE_RELU_1D_ARRAY_VALUE(float, Float)
TEST_VALUE_RELU_1D_ARRAY_VALUE(double, Double)
TEST_VALUE_RELU_1D_ARRAY_VALUE(int, Int)


#define TEST_VALUE_RELU_2D_ARRAY_VALUE(TYPE, NAME)                            \
    TEST(ValueTest, Relu##NAME##2DArrayValue) {                               \
        std::vector<std::vector<Value<TYPE>>> a = {                           \
            {Value<TYPE>(2.0), Value<TYPE>(-3.0), Value<TYPE>(4.0)},          \
            {Value<TYPE>(5.0), Value<TYPE>(6.0), Value<TYPE>(-7.0)}           \
        };                                                                    \
                                                                              \
        std::vector<std::vector<Value<TYPE>>> b = ptMgrad::relu(a);           \
                                                                              \
        EXPECT_EQ(b[0][0].dataX(), TYPE(2.0));                                \
        EXPECT_EQ(b[0][1].dataX(), TYPE(0.0));                                \
        EXPECT_EQ(b[0][2].dataX(), TYPE(4.0));                                \
        EXPECT_EQ(b[1][0].dataX(), TYPE(5.0));                                \
        EXPECT_EQ(b[1][1].dataX(), TYPE(6.0));                                \
        EXPECT_EQ(b[1][2].dataX(), TYPE(0.0));                                \
    }

TEST_VALUE_RELU_2D_ARRAY_VALUE(float, Float)
TEST_VALUE_RELU_2D_ARRAY_VALUE(double, Double)
TEST_VALUE_RELU_2D_ARRAY_VALUE(int, Int)
