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

#define TEST_VALUE_ADD(TYPE, NAME)                       \
    TEST(ValueTest, Add##NAME) {                         \
        Value<TYPE> a = 2.0;                             \
        Value<TYPE> b = 3.0;                             \
                                                         \
        Value<TYPE> c = ptMgrad::add(a, b);              \
        c.backward();                                    \
                                                         \
        EXPECT_EQ(c.dataX(), 5.0);                       \
        EXPECT_EQ(a.gradX(), 1.0);                       \
        EXPECT_EQ(b.gradX(), 1.0);                       \
        a.zero_grad();                                   \
        b.zero_grad();                                   \
    }

TEST_VALUE_ADD(float, Float)
TEST_VALUE_ADD(double, Double)
TEST_VALUE_ADD(int, Int)


#define TEST_VALUE_ADD_SCALAR(TYPE, NAME)                \
    TEST(ValueTest, Add##NAME##Scalar) {                 \
        Value<TYPE> a = 2.0;                             \
        TYPE b = 3.0;                                    \
                                                         \
        Value<TYPE> c = ptMgrad::add(a, b);              \
        c.backward();                                    \
                                                         \
        EXPECT_EQ(c.dataX(), 5.0);                       \
        EXPECT_EQ(a.gradX(), 1.0);                       \
        a.zero_grad();                                   \
    }

TEST_VALUE_ADD_SCALAR(float, Float)
TEST_VALUE_ADD_SCALAR(double, Double)
TEST_VALUE_ADD_SCALAR(int, Int)


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


#define TEST_VALUE_ADD_SCALAR_SCALAR(TYPE, NAME)            \
    TEST(ValueTest, ScalarSCALARADD##NAME) {                \
        TYPE a = 2.0;                                       \
        TYPE b = 3.0;                                       \
                                                            \
        Value<TYPE> c = ptMgrad::add(a, b);                 \
                                                            \
        EXPECT_EQ(c.dataX(), 5.0);                          \
    }

TEST_VALUE_ADD_SCALAR_SCALAR(float, Float)
TEST_VALUE_ADD_SCALAR_SCALAR(double, Double)
TEST_VALUE_ADD_SCALAR_SCALAR(int, Int)


// TODO: add backward support; investigate
#define TEST_VALUE_ADD_VECTOR(TYPE, NAME)                        \
    TEST(ValueTest, Add##NAME##Vector) {                         \
        std::vector<Value<TYPE>> a = {2.0, 3.0, 4.0};            \
        std::vector<Value<TYPE>> b = {5.0, -6.0, 7.0};           \
                                                                 \
        std::vector<Value<TYPE>> c = ptMgrad::add(a, b);         \
                                                                 \
        EXPECT_EQ(c[0].dataX(), 7.0);                            \
        EXPECT_EQ(c[1].dataX(), -3.0);                           \
        EXPECT_EQ(c[2].dataX(), 11.0);                           \
    }

/*
        for (auto &v : c) {                                     \
            v.backward();                                       \
        }                                                       \
                                                                \
        EXPECT_EQ(c[0].dataX(), 7.0);                           \
        EXPECT_EQ(c[1].dataX(), -3.0);                          \
        EXPECT_EQ(c[2].dataX(), 11.0);                          \
                                                                \
        EXPECT_EQ(a[0].gradX(), 1.0);                           \
        EXPECT_EQ(a[1].gradX(), 1.0);                           \
        EXPECT_EQ(a[2].gradX(), 1.0);                           \
                                                                \
        EXPECT_EQ(b[0].gradX(), 1.0);                           \
        EXPECT_EQ(b[1].gradX(), -1.0);                          \
        EXPECT_EQ(b[2].gradX(), 1.0);                           \
                                                                \
        for (auto &v : a) {                                     \
            v.zero_grad();                                      \
        }                                                       \
                                                                \
        for (auto &v : b) {                                     \
            v.zero_grad();                                      \
        }                                                       \
    }
*/

TEST_VALUE_ADD_VECTOR(float, Float)
TEST_VALUE_ADD_VECTOR(double, Double)
TEST_VALUE_ADD_VECTOR(int, Int)


#define TEST_VALUE_ADD_VECTOR_SCALAR(TYPE, NAME)                 \
    TEST(ValueTest, Add##NAME##VectorScalar) {                   \
        std::vector<Value<TYPE>> a = {2.0, 3.0, 4.0};            \
        TYPE b = 5.0;                                            \
                                                                 \
        std::vector<Value<TYPE>> c = ptMgrad::add(a, b);         \
                                                                 \
        EXPECT_EQ(c[0].dataX(), 7.0);                            \
        EXPECT_EQ(c[1].dataX(), 8.0);                            \
        EXPECT_EQ(c[2].dataX(), 9.0);                            \
    }

TEST_VALUE_ADD_VECTOR_SCALAR(float, Float)
TEST_VALUE_ADD_VECTOR_SCALAR(double, Double)
TEST_VALUE_ADD_VECTOR_SCALAR(int, Int)


#define TEST_VALUE_ADD_MATRIX(TYPE, NAME)                                   \
    TEST(ValueTest, Add##NAME##Matrix) {                                    \
        std::vector<std::vector<Value<TYPE>>> a = {                         \
            {2.0, 3.0, 4.0},                                                \
            {5.0, -6.0, 7.0}                                                \
        };                                                                  \
        std::vector<std::vector<Value<TYPE>>> b = {                         \
            {5.0, -6.0, 7.0},                                               \
            {2.0, 3.0, 4.0}                                                 \
        };                                                                  \
                                                                            \
        std::vector<std::vector<Value<TYPE>>> c = ptMgrad::add(a, b);       \
                                                                            \
        EXPECT_EQ(c[0][0].dataX(), 7.0);                                    \
        EXPECT_EQ(c[0][1].dataX(), -3.0);                                   \
        EXPECT_EQ(c[0][2].dataX(), 11.0);                                   \
        EXPECT_EQ(c[1][0].dataX(), 7.0);                                    \
        EXPECT_EQ(c[1][1].dataX(), -3.0);                                   \
        EXPECT_EQ(c[1][2].dataX(), 11.0);                                   \
    }

TEST_VALUE_ADD_MATRIX(float, Float)
TEST_VALUE_ADD_MATRIX(double, Double)
TEST_VALUE_ADD_MATRIX(int, Int)


#define TEST_VALUE_ADD_MATRIX_SCALAR(TYPE, NAME)                              \
    TEST(ValueTest, Add##NAME##MatrixScalar) {                                \
        std::vector<std::vector<Value<TYPE>>> a = {                           \
            {2.0, 3.0, 4.0},                                                  \
            {5.0, -6.0, 7.0}                                                  \
        };                                                                    \
        TYPE b = 5.0;                                                         \
                                                                              \
        std::vector<std::vector<Value<TYPE>>> c = ptMgrad::add(a, b);         \
                                                                              \
        EXPECT_EQ(c[0][0].dataX(), 7.0);                                      \
        EXPECT_EQ(c[0][1].dataX(), 8.0);                                      \
        EXPECT_EQ(c[0][2].dataX(), 9.0);                                      \
        EXPECT_EQ(c[1][0].dataX(), 10.0);                                     \
        EXPECT_EQ(c[1][1].dataX(), -1.0);                                     \
        EXPECT_EQ(c[1][2].dataX(), 12.0);                                     \
    }

TEST_VALUE_ADD_MATRIX_SCALAR(float, Float)
TEST_VALUE_ADD_MATRIX_SCALAR(double, Double)
TEST_VALUE_ADD_MATRIX_SCALAR(int, Int)


// substraction

#define TEST_VALUE_SUB(TYPE, NAME)                   \
    TEST(ValueTest, Sub##NAME) {                     \
        Value<TYPE> a = 2.0;                         \
        Value<TYPE> b = 3.0;                         \
                                                     \
        Value<TYPE> c = ptMgrad::sub(a, b);          \
        c.backward();                                \
                                                     \
        EXPECT_EQ(c.dataX(), -1.0);                  \
        EXPECT_EQ(a.gradX(), 1.0);                   \
        EXPECT_EQ(b.gradX(), -1.0);                  \
        a.zero_grad();                               \
        b.zero_grad();                               \
    }

TEST_VALUE_SUB(float, Float)
TEST_VALUE_SUB(double, Double)
TEST_VALUE_SUB(int, Int)


#define TEST_VALUE_SUB_SCALAR(TYPE, NAME)                \
    TEST(ValueTest, Sub##NAME##Scalar) {                 \
        Value<TYPE> a = 2.0;                             \
        TYPE b = 3.0;                                    \
                                                         \
        Value<TYPE> c = ptMgrad::sub(a, b);              \
        c.backward();                                    \
                                                         \
        EXPECT_EQ(c.dataX(), -1.0);                      \
        EXPECT_EQ(a.gradX(), 1.0);                       \
        a.zero_grad();                                   \
    }

TEST_VALUE_SUB_SCALAR(float, Float)
TEST_VALUE_SUB_SCALAR(double, Double)
TEST_VALUE_SUB_SCALAR(int, Int)


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


#define TEST_VALUE_SUB_SCALAR_SCALAR(TYPE, NAME)            \
    TEST(ValueTest, Sub##NAME##ScalarScalar) {              \
        TYPE a = 2.0;                                       \
        TYPE b = 3.0;                                       \
                                                            \
        Value<TYPE> c = ptMgrad::sub(a, b);                 \
                                                            \
        EXPECT_EQ(c.dataX(), -1.0);                         \
    }

TEST_VALUE_SUB_SCALAR_SCALAR(float, Float)
TEST_VALUE_SUB_SCALAR_SCALAR(double, Double)
TEST_VALUE_SUB_SCALAR_SCALAR(int, Int)


// for vector x vector operations
TEST(ValueTest, VectorSub) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    std::vector<Value<float>> b = {5.0f, -6.0f, 7.0f};

    std::vector<Value<float>> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c[0].dataX(), -3.0f);
    EXPECT_EQ(c[1].dataX(), 9.0f);
    EXPECT_EQ(c[2].dataX(), -3.0f);
}


// for vector x scalar operations
TEST(ValueTest, VectorSubScalar) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    float b = 5.0f;

    std::vector<Value<float>> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c[0].dataX(), -3.0f);
    EXPECT_EQ(c[1].dataX(), -2.0f);
    EXPECT_EQ(c[2].dataX(), -1.0f);
}


// for matrix x matrix operations
TEST(ValueTest, MatrixSub) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    std::vector<std::vector<Value<float>>> b = {
        {5.0f, -6.0f, 7.0f},
        {2.0f, 3.0f, 4.0f}
    };

    std::vector<std::vector<Value<float>>> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c[0][0].dataX(), -3.0f);
    EXPECT_EQ(c[0][1].dataX(), 9.0f);
    EXPECT_EQ(c[0][2].dataX(), -3.0f);
    EXPECT_EQ(c[1][0].dataX(), 3.0f);
    EXPECT_EQ(c[1][1].dataX(), -9.0f);
    EXPECT_EQ(c[1][2].dataX(), 3.0f);
}


// for matrix x scalar operations
TEST(ValueTest, MatrixSubScalar) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    float b = 5.0f;

    std::vector<std::vector<Value<float>>> c = ptMgrad::sub(a, b);

    EXPECT_EQ(c[0][0].dataX(), -3.0f);
    EXPECT_EQ(c[0][1].dataX(), -2.0f);
    EXPECT_EQ(c[0][2].dataX(), -1.0f);
    EXPECT_EQ(c[1][0].dataX(), 0.0f);
    EXPECT_EQ(c[1][1].dataX(), -11.0f);
    EXPECT_EQ(c[1][2].dataX(), 2.0f);
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

/*
// for vector x scalar operations
TEST(ValueTest, VectorRsubScalar) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    float b = 5.0f;

    std::vector<Value<float>> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c[0].dataX(), 3.0f);
    EXPECT_EQ(c[1].dataX(), 2.0f);
    EXPECT_EQ(c[2].dataX(), 1.0f);
}

// for matrix x scalar operations
TEST(ValueTest, MatrixRsubScalar) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    float b = 5.0f;

    std::vector<std::vector<Value<float>>> c = ptMgrad::rsub(a, b);

    EXPECT_EQ(c[0][0].dataX(), 3.0f);
    EXPECT_EQ(c[0][1].dataX(), 2.0f);
    EXPECT_EQ(c[0][2].dataX(), 1.0f);
    EXPECT_EQ(c[1][0].dataX(), 0.0f);
    EXPECT_EQ(c[1][1].dataX(), 11.0f);
    EXPECT_EQ(c[1][2].dataX(), -2.0f);
}
*/

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

// for vector x scalar operations
TEST(ValueTest, VectorMulScalar) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    float b = 5.0f;

    std::vector<Value<float>> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c[0].dataX(), 10.0f);
    EXPECT_EQ(c[1].dataX(), 15.0f);
    EXPECT_EQ(c[2].dataX(), 20.0f);
}

// for vector x vector operations
TEST(ValueTest, VectorMulVector) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    std::vector<Value<float>> b = {5.0f, -6.0f, 7.0f};

    std::vector<Value<float>> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c[0].dataX(), 10.0f);
    EXPECT_EQ(c[1].dataX(), -18.0f);
    EXPECT_EQ(c[2].dataX(), 28.0f);
}
/*
// for matrix x vector operations
TEST(ValueTest, MatrixMulVector) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    std::vector<Value<float>> b = {5.0f, -6.0f, 7.0f};

    std::vector<std::vector<Value<float>>> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c[0][0].dataX(), 10.0f);
    EXPECT_EQ(c[0][1].dataX(), -18.0f);
    EXPECT_EQ(c[0][2].dataX(), 28.0f);
    EXPECT_EQ(c[1][0].dataX(), 10.0f);
    EXPECT_EQ(c[1][1].dataX(), -18.0f);
    EXPECT_EQ(c[1][2].dataX(), 28.0f);
}
*/
// for matrix x matrix operations
TEST(ValueTest, MatrixMul) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    std::vector<std::vector<Value<float>>> b = {
        {5.0f, -6.0f, 7.0f},
        {2.0f, 3.0f, 4.0f}
    };

    std::vector<std::vector<Value<float>>> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c[0][0].dataX(), 10.0f);
    EXPECT_EQ(c[0][1].dataX(), -18.0f);
    EXPECT_EQ(c[0][2].dataX(), 28.0f);
    EXPECT_EQ(c[1][0].dataX(), 10.0f);
    EXPECT_EQ(c[1][1].dataX(), -18.0f);
    EXPECT_EQ(c[1][2].dataX(), 28.0f);
}

// for matrix x scalar operations
TEST(ValueTest, MatrixMulScalar) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    float b = 5.0f;

    std::vector<std::vector<Value<float>>> c = ptMgrad::mul(a, b);

    EXPECT_EQ(c[0][0].dataX(), 10.0f);
    EXPECT_EQ(c[0][1].dataX(), 15.0f);
    EXPECT_EQ(c[0][2].dataX(), 20.0f);
    EXPECT_EQ(c[1][0].dataX(), 25.0f);
    EXPECT_EQ(c[1][1].dataX(), -30.0f);
    EXPECT_EQ(c[1][2].dataX(), 35.0f);
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

// for vector x vector operations
TEST(ValueTest, VectorDiv) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    std::vector<Value<float>> b = {5.0f, -6.0f, 7.0f};

    std::vector<Value<float>> c = ptMgrad::div(a, b);

    EXPECT_EQ(c[0].dataX(), 2.0f / 5.0f);
    EXPECT_EQ(c[1].dataX(), 3.0f / -6.0f);
    EXPECT_EQ(c[2].dataX(), 4.0f / 7.0f);
}

// for vector x scalar operations
TEST(ValueTest, VectorDivScalar) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    float b = 5.0f;

    std::vector<Value<float>> c = ptMgrad::div(a, b);

    EXPECT_EQ(c[0].dataX(), 2.0f / 5.0f);
    EXPECT_EQ(c[1].dataX(), 3.0f / 5.0f);
    EXPECT_EQ(c[2].dataX(), 4.0f / 5.0f);
}

// for matrix x matrix operations
TEST(ValueTest, MatrixDiv) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    std::vector<std::vector<Value<float>>> b = {
        {5.0f, -6.0f, 7.0f},
        {2.0f, 3.0f, 4.0f}
    };

    std::vector<std::vector<Value<float>>> c = ptMgrad::div(a, b);

    EXPECT_EQ(c[0][0].dataX(), 2.0f / 5.0f);
    EXPECT_EQ(c[0][1].dataX(), 3.0f / -6.0f);
    EXPECT_EQ(c[0][2].dataX(), 4.0f / 7.0f);
    EXPECT_EQ(c[1][0].dataX(), 5.0f / 2.0f);
    EXPECT_EQ(c[1][1].dataX(), -6.0f / 3.0f);
    EXPECT_EQ(c[1][2].dataX(), 7.0f / 4.0f);
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

// for vector x vector operations
TEST(ValueTest, VectorRdiv) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    std::vector<Value<float>> b = {5.0f, -6.0f, 7.0f};

    std::vector<Value<float>> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c[0].dataX(), 5.0f / 2.0f);
    EXPECT_EQ(c[1].dataX(), -6.0f / 3.0f);
    EXPECT_EQ(c[2].dataX(), 7.0f / 4.0f);
}
/*
// for vector x scalar operations
TEST(ValueTest, VectorRdivScalar) {
    std::vector<Value<float>> a = {2.0f, 3.0f, 4.0f};
    float b = 5.0f;

    std::vector<Value<float>> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c[0].dataX(), 5.0f / 2.0f);
    EXPECT_EQ(c[1].dataX(), 5.0f / 3.0f);
    EXPECT_EQ(c[2].dataX(), 5.0f / 4.0f);
}

// for matrix x matrix operations
TEST(ValueTest, MatrixRdiv) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    std::vector<std::vector<Value<float>>> b = {
        {5.0f, -6.0f, 7.0f},
        {2.0f, 3.0f, 4.0f}
    };

    std::vector<std::vector<Value<float>>> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c[0][0].dataX(), 5.0f / 2.0f);
    EXPECT_EQ(c[0][1].dataX(), -6.0f / 3.0f);
    EXPECT_EQ(c[0][2].dataX(), 7.0f / 4.0f);
    EXPECT_EQ(c[1][0].dataX(), 2.0f / 5.0f);
    EXPECT_EQ(c[1][1].dataX(), 3.0f / -6.0f);
    EXPECT_EQ(c[1][2].dataX(), 4.0f / 7.0f);
}

// matrix x scalar operations
TEST(ValueTest, MatrixRdivScalar) {
    std::vector<std::vector<Value<float>>> a = {
        {2.0f, 3.0f, 4.0f},
        {5.0f, -6.0f, 7.0f}
    };
    float b = 5.0f;

    std::vector<std::vector<Value<float>>> c = ptMgrad::rdiv(a, b);

    EXPECT_EQ(c[0][0].dataX(), 5.0f / 2.0f);
    EXPECT_EQ(c[0][1].dataX(), 5.0f / 3.0f);
    EXPECT_EQ(c[0][2].dataX(), 5.0f / 4.0f);
    EXPECT_EQ(c[1][0].dataX(), 5.0f / 5.0f);
    EXPECT_EQ(c[1][1].dataX(), 5.0f / -6.0f);
    EXPECT_EQ(c[1][2].dataX(), 5.0f / 7.0f);
}
*/

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
