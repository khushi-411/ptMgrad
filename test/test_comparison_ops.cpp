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

#define TEST_VALUE_LT(TYPE, NAME)                    \
    TEST(ValueTest, Lt##NAME) {                      \
        Value<TYPE> a = 2.0;                         \
        Value<TYPE> b = 3.0;                         \
                                                     \
        bool c = ptMgrad::lt(a, b);                  \
                                                     \
        EXPECT_EQ(c, true);                          \
    }

TEST_VALUE_LT(float, Float)
TEST_VALUE_LT(double, Double)
TEST_VALUE_LT(int, Int)


#define TEST_VALUE_LT_SCALAR(TYPE, NAME)             \
    TEST(ValueTest, Lt##NAME##Scalar) {              \
        Value<TYPE> a = 2.0;                         \
        TYPE b = 3.0;                                \
                                                     \
        bool c = ptMgrad::lt(a, b);                  \
                                                     \
        EXPECT_EQ(c, true);                          \
    }

TEST_VALUE_LT_SCALAR(float, Float)
TEST_VALUE_LT_SCALAR(double, Double)
TEST_VALUE_LT_SCALAR(int, Int)


#define TEST_VALUE_LT_SCALAR_SCALAR(TYPE, NAME)       \
    TEST(ValueTest, Lt##NAME##ScalarScalar) {         \
        TYPE a = 2.0;                                 \
        TYPE b = 3.0;                                 \
                                                      \
        bool c = ptMgrad::lt(a, b);                   \
                                                      \
        EXPECT_EQ(c, true);                           \
    }

TEST_VALUE_LT_SCALAR_SCALAR(float, Float)
TEST_VALUE_LT_SCALAR_SCALAR(double, Double)
TEST_VALUE_LT_SCALAR_SCALAR(int, Int)


#define TEST_VALUE_LT_VECTOR_VECTOR(TYPE, NAME)             \
    TEST(ValueTest, Lt##NAME##VectorVector) {               \
        std::vector<Value<TYPE>> a = {2.0, -3.0, 4.0};      \
                                                            \
        std::vector<Value<TYPE>> b = {5.0, 6.0, -7.0};      \
                                                            \
        std::vector<bool> c = ptMgrad::lt(a, b);            \
                                                            \
        EXPECT_EQ(c[0], true);                              \
        EXPECT_EQ(c[1], true);                              \
        EXPECT_EQ(c[2], false);                             \
    }

TEST_VALUE_LT_VECTOR_VECTOR(float, Float)
TEST_VALUE_LT_VECTOR_VECTOR(double, Double)
TEST_VALUE_LT_VECTOR_VECTOR(int, Int)


#define TEST_VALUE_LT_VECTOR_SCALAR(TYPE, NAME)             \
    TEST(ValueTest, Lt##NAME##VectorScalar) {               \
        std::vector<Value<TYPE>> a = {2.0, -3.0, 4.0};      \
                                                            \
        TYPE b = 5.0;                                       \
                                                            \
        std::vector<bool> c = ptMgrad::lt(a, b);            \
                                                            \
        EXPECT_EQ(c[0], true);                              \
        EXPECT_EQ(c[1], true);                              \
        EXPECT_EQ(c[2], true);                              \
    }

TEST_VALUE_LT_VECTOR_SCALAR(float, Float)
TEST_VALUE_LT_VECTOR_SCALAR(double, Double)
TEST_VALUE_LT_VECTOR_SCALAR(int, Int)


#define TEST_VALUE_LT_MATRIX_MATRIX(TYPE, NAME)                   \
    TEST(ValueTest, Lt##NAME##MatrixMatrix) {                     \
        std::vector<std::vector<Value<TYPE>>> a = {               \
            {2.0, 3.0, 4.0},                                      \
            {5.0, -6.0, 7.0}                                      \
        };                                                        \
                                                                  \
        std::vector<std::vector<Value<TYPE>>> b = {               \
            {5.0, 6.0, -7.0},                                     \
            {2.0, 3.0, 4.0}                                       \
        };                                                        \
                                                                  \
        std::vector<std::vector<bool>> c = ptMgrad::lt(a, b);     \
                                                                  \
        EXPECT_EQ(c[0][0], true);                                 \
        EXPECT_EQ(c[0][1], true);                                 \
        EXPECT_EQ(c[0][2], false);                                \
        EXPECT_EQ(c[1][0], false);                                \
        EXPECT_EQ(c[1][1], true);                                 \
        EXPECT_EQ(c[1][2], false);                                \
    }

TEST_VALUE_LT_MATRIX_MATRIX(float, Float)
TEST_VALUE_LT_MATRIX_MATRIX(double, Double)
TEST_VALUE_LT_MATRIX_MATRIX(int, Int)


#define TEST_VALUE_LT_MATRIX_MATRIX_VALUE(TYPE, NAME)                   \
    TEST(ValueTest, Lt##NAME##MatrixMatrixValue) {                      \
        std::vector<std::vector<Value<TYPE>>> a = {                     \
            {Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(4.0)},     \
            {Value<TYPE>(5.0), Value<TYPE>(-6.0), Value<TYPE>(7.0)}     \
        };                                                              \
                                                                        \
        std::vector<std::vector<Value<TYPE>>> b = {                     \
            {Value<TYPE>(0.0), Value<TYPE>(6.0), Value<TYPE>(-7.0)},    \
            {Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(4.0)}      \
        };                                                              \
                                                                        \
        std::vector<std::vector<bool>> c = ptMgrad::lt(a, b);           \
                                                                        \
        EXPECT_EQ(c[0][0], false);                                      \
        EXPECT_EQ(c[0][1], true);                                       \
        EXPECT_EQ(c[0][2], false);                                      \
        EXPECT_EQ(c[1][0], false);                                      \
        EXPECT_EQ(c[1][1], true);                                       \
        EXPECT_EQ(c[1][2], false);                                      \
    }

TEST_VALUE_LT_MATRIX_MATRIX_VALUE(float, Float)
TEST_VALUE_LT_MATRIX_MATRIX_VALUE(double, Double)
TEST_VALUE_LT_MATRIX_MATRIX_VALUE(int, Int)


#define TEST_VALUE_LT_MATRIX_SCALAR(TYPE, NAME)                    \
    TEST(ValueTest, Lt##NAME##MatrixScalar) {                      \
        std::vector<std::vector<Value<TYPE>>> a = {                \
            {2.0, 3.0, 4.0},                                       \
            {5.0, -6.0, 7.0}                                       \
        };                                                         \
                                                                   \
        TYPE b = 5.0;                                              \
                                                                   \
        std::vector<std::vector<bool>> c = ptMgrad::lt(a, b);      \
                                                                   \
        EXPECT_EQ(c[0][0], true);                                  \
        EXPECT_EQ(c[0][1], true);                                  \
        EXPECT_EQ(c[0][2], true);                                  \
        EXPECT_EQ(c[1][0], false);                                 \
        EXPECT_EQ(c[1][1], true);                                  \
        EXPECT_EQ(c[1][2], false);                                 \
    }

TEST_VALUE_LT_MATRIX_SCALAR(float, Float)
TEST_VALUE_LT_MATRIX_SCALAR(double, Double)
TEST_VALUE_LT_MATRIX_SCALAR(int, Int)


// gt

#define TEST_VALUE_GT(TYPE, NAME)                    \
    TEST(ValueTest, Gt##NAME) {                      \
        Value<TYPE> a = 2.0;                         \
        Value<TYPE> b = 3.0;                         \
                                                     \
        bool c = ptMgrad::gt(a, b);                  \
                                                     \
        EXPECT_EQ(c, false);                         \
    }

TEST_VALUE_GT(float, Float)
TEST_VALUE_GT(double, Double)
TEST_VALUE_GT(int, Int)


#define TEST_VALUE_GT_VALUE_SCALAR(TYPE, NAME)       \
    TEST(ValueTest, Gt##NAME##Scalar) {              \
        Value<TYPE> a = 2.0;                         \
        TYPE b = 3.0;                                \
                                                     \
        bool c = ptMgrad::gt(a, b);                  \
                                                     \
        EXPECT_EQ(c, false);                         \
    }

TEST_VALUE_GT_VALUE_SCALAR(float, Float)
TEST_VALUE_GT_VALUE_SCALAR(double, Double)
TEST_VALUE_GT_VALUE_SCALAR(int, Int)


#define TEST_VALUE_GT_SCALAR_SCALAR(TYPE, NAME)       \
    TEST(ValueTest, Gt##NAME##ScalarScalar) {         \
        TYPE a = -2.0;                                \
        TYPE b = 3.0;                                 \
                                                      \
        bool c = ptMgrad::gt(a, b);                   \
                                                      \
        EXPECT_EQ(c, false);                          \
    }

TEST_VALUE_GT_SCALAR_SCALAR(float, Float)
TEST_VALUE_GT_SCALAR_SCALAR(double, Double)
TEST_VALUE_GT_SCALAR_SCALAR(int, Int)


#define TEST_VALUE_GT_VECTOR_VECTOR(TYPE, NAME)                       \
    TEST(ValueTest, Gt##NAME##VectorVector) {                         \
        std::vector<Value<TYPE>> a = {                                \
            Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(4.0)      \
        };                                                            \
                                                                      \
        std::vector<Value<TYPE>> b = {                                \
            Value<TYPE>(5.0), Value<TYPE>(-6.0), Value<TYPE>(7.0)     \
        };                                                            \
                                                                      \
        std::vector<bool> c = ptMgrad::gt(a, b);                      \
                                                                      \
        EXPECT_EQ(c[0], false);                                       \
        EXPECT_EQ(c[1], true);                                        \
        EXPECT_EQ(c[2], false);                                       \
    }

TEST_VALUE_GT_VECTOR_VECTOR(float, Float)
TEST_VALUE_GT_VECTOR_VECTOR(double, Double)
TEST_VALUE_GT_VECTOR_VECTOR(int, Int)


#define TEST_VALUE_GT_VECTOR_VALUE(TYPE, NAME)                        \
    TEST(ValueTest, Gt##NAME##VectorValue) {                          \
        std::vector<Value<TYPE>> a = {                                \
            Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(4.0)      \
        };                                                            \
                                                                      \
        std::vector<Value<TYPE>> b = {5.0, -6.0, 7.0};                \
                                                                      \
        std::vector<bool> c = ptMgrad::gt(a, b);                      \
                                                                      \
        EXPECT_EQ(c[0], false);                                       \
        EXPECT_EQ(c[1], true);                                        \
        EXPECT_EQ(c[2], false);                                       \
    }

TEST_VALUE_GT_VECTOR_VALUE(float, Float)
TEST_VALUE_GT_VECTOR_VALUE(double, Double)
TEST_VALUE_GT_VECTOR_VALUE(int, Int)


#define TEST_VALUE_GT_VECTOR_SCALAR(TYPE, NAME)                       \
    TEST(ValueTest, Gt##NAME##VectorScalar) {                         \
        std::vector<Value<TYPE>> a = {                                \
            Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(8.0)      \
        };                                                            \
                                                                      \
        TYPE b =  5.0;                                                \
                                                                      \
        std::vector<bool> c = ptMgrad::gt(a, b);                      \
                                                                      \
        EXPECT_EQ(c[0], false);                                       \
        EXPECT_EQ(c[1], false);                                       \
        EXPECT_EQ(c[2], true);                                        \
    }

TEST_VALUE_GT_VECTOR_SCALAR(float, Float)
TEST_VALUE_GT_VECTOR_SCALAR(double, Double)
TEST_VALUE_GT_VECTOR_SCALAR(int, Int)


#define TEST_VALUE_GT_VALUE_SCALAR_SCALAR(TYPE, NAME)                 \
    TEST(ValueTest, Gt##NAME##ValueScalarScalar) {                    \
        std::vector<Value<TYPE>> a = {2.0, 3.0, 4.0};                 \
                                                                      \
        TYPE b = 5.0;                                                 \
                                                                      \
        std::vector<bool> c = ptMgrad::gt(a, b);                      \
                                                                      \
        EXPECT_EQ(c[0], false);                                       \
        EXPECT_EQ(c[1], false);                                       \
        EXPECT_EQ(c[2], false);                                       \
    }

TEST_VALUE_GT_VALUE_SCALAR_SCALAR(float, Float)
TEST_VALUE_GT_VALUE_SCALAR_SCALAR(double, Double)
TEST_VALUE_GT_VALUE_SCALAR_SCALAR(int, Int)


#define TEST_VALUE_GT_MATRIX_MATRIX(TYPE, NAME)                         \
    TEST(ValueTest, Gt##NAME##MatrixMatrix) {                           \
        std::vector<std::vector<Value<TYPE>>> a = {                     \
            {Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(4.0)},     \
            {Value<TYPE>(5.0), Value<TYPE>(-6.0), Value<TYPE>(7.0)}     \
        };                                                              \
                                                                        \
        std::vector<std::vector<Value<TYPE>>> b = {                     \
            {Value<TYPE>(5.0), Value<TYPE>(-6.0), Value<TYPE>(7.0)},    \
            {Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(4.0)}      \
        };                                                              \
                                                                        \
        std::vector<std::vector<bool>> c = ptMgrad::gt(a, b);           \
                                                                        \
        EXPECT_EQ(c[0][0], false);                                      \
        EXPECT_EQ(c[0][1], true);                                       \
        EXPECT_EQ(c[0][2], false);                                      \
        EXPECT_EQ(c[1][0], true);                                       \
        EXPECT_EQ(c[1][1], false);                                      \
        EXPECT_EQ(c[1][2], true);                                       \
    }

TEST_VALUE_GT_MATRIX_MATRIX(float, Float)
TEST_VALUE_GT_MATRIX_MATRIX(double, Double)
TEST_VALUE_GT_MATRIX_MATRIX(int, Int)


#define TEST_VALUE_GT_MATRIX_SCALAR(TYPE, NAME)                         \
    TEST(ValueTest, Gt##NAME##MatrixScalar) {                           \
        std::vector<std::vector<Value<TYPE>>> a = {                     \
            {Value<TYPE>(2.0), Value<TYPE>(3.0), Value<TYPE>(4.0)},     \
            {Value<TYPE>(5.0), Value<TYPE>(-6.0), Value<TYPE>(7.0)}     \
        };                                                              \
                                                                        \
        TYPE b = 5.0;                                                   \
                                                                        \
        std::vector<std::vector<bool>> c = ptMgrad::gt(a, b);           \
                                                                        \
        EXPECT_EQ(c[0][0], false);                                      \
        EXPECT_EQ(c[0][1], false);                                      \
        EXPECT_EQ(c[0][2], false);                                      \
        EXPECT_EQ(c[1][0], false);                                      \
        EXPECT_EQ(c[1][1], false);                                      \
        EXPECT_EQ(c[1][2], true);                                       \
    }

TEST_VALUE_GT_MATRIX_SCALAR(float, Float)
TEST_VALUE_GT_MATRIX_SCALAR(double, Double)
TEST_VALUE_GT_MATRIX_SCALAR(int, Int)
