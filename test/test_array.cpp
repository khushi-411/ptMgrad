#include <iostream>

#include <gtest/gtest.h>

#include "../src/engine.h"
#include "../src/Array.h"


using namespace ptMgrad;


TEST(ValueTest, FloatArray) {
    Array<Value<float>> a;
    a.push_back(2.0f);
    a.push_back(3.0f);
    a.push_back(4.0f);

    EXPECT_EQ(a[0].dataX(), 2.0f);
    EXPECT_EQ(a[1].dataX(), 3.0f);
    EXPECT_EQ(a[2].dataX(), 4.0f);
}

#define TEST_ARRAY(TYPE, NAME)                   \
    TEST(ValueTest, Array##NAME) {               \
        Array<Value<TYPE>> a;                    \
        a.push_back(TYPE(2.0));                  \
        a.push_back(TYPE(3.0));                  \
        a.push_back(TYPE(4.0));                  \
                                                 \
        EXPECT_EQ(a[0].dataX(), TYPE(2.0));      \
        EXPECT_EQ(a[1].dataX(), TYPE(3.0));      \
        EXPECT_EQ(a[2].dataX(), TYPE(4.0));      \
    }

TEST_ARRAY(float, Float)
TEST_ARRAY(double, Double)
TEST_ARRAY(int, Int)
TEST_ARRAY(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_INITIALIZELIST(TYPE, NAME)                         \
    TEST(ValueTest, Array##NAME##Initialize) {                        \
        Array<Value<TYPE>> a = {TYPE(2.0), TYPE(3.0), TYPE(4.0)};     \
                                                                      \
        EXPECT_EQ(a[0].dataX(), TYPE(2.0));                           \
        EXPECT_EQ(a[1].dataX(), TYPE(3.0));                           \
        EXPECT_EQ(a[2].dataX(), TYPE(4.0));                           \
    }

TEST_ARRAY_INITIALIZELIST(float, Float)
TEST_ARRAY_INITIALIZELIST(double, Double)
TEST_ARRAY_INITIALIZELIST(int, Int)
TEST_ARRAY_INITIALIZELIST(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_INITIALIZELIST(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_SIZE(TYPE, NAME)                      \
    TEST(ValueTest, Array##NAME##Size) {                 \
        Array<Value<TYPE>> a;                            \
        a.push_back(TYPE(2.0));                          \
        a.push_back(TYPE(3.0));                          \
        a.push_back(TYPE(4.0));                          \
                                                         \
        EXPECT_EQ(a.size(), 3);                          \
    }

TEST_ARRAY_SIZE(float, Float)
TEST_ARRAY_SIZE(double, Double)
TEST_ARRAY_SIZE(int, Int)
TEST_ARRAY_SIZE(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_SIZE(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_CLEAR(TYPE, NAME)                      \
    TEST(ValueTest, Array##NAME##Clear) {                 \
        Array<Value<TYPE>> a;                             \
        a.push_back(TYPE(2.0));                           \
        a.push_back(TYPE(3.0));                           \
        a.push_back(TYPE(4.0));                           \
                                                          \
        a.clear();                                        \
                                                          \
        EXPECT_EQ(a.size(), 0);                           \
    }

TEST_ARRAY_CLEAR(float, Float)
TEST_ARRAY_CLEAR(double, Double)
TEST_ARRAY_CLEAR(int, Int)
TEST_ARRAY_CLEAR(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_CLEAR(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_POPBACK(TYPE, NAME)                   \
    TEST(ValueTest, Array##NAME##PopBack) {              \
        Array<Value<TYPE>> a;                            \
        a.push_back(TYPE(2.0));                          \
        a.push_back(TYPE(3.0));                          \
        a.push_back(TYPE(4.0));                          \
                                                         \
        a.pop_back();                                    \
                                                         \
        EXPECT_EQ(a.size(), 2);                          \
    }

TEST_ARRAY_POPBACK(float, Float)
TEST_ARRAY_POPBACK(double, Double)
TEST_ARRAY_POPBACK(int, Int)
TEST_ARRAY_POPBACK(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_POPBACK(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_EMPTY(TYPE, NAME)                     \
    TEST(ValueTest, Array##NAME##Empty) {                \
        Array<Value<TYPE>> a;                            \
        EXPECT_EQ(a.empty(), true);                      \
                                                         \
        a.push_back(TYPE(2.0));                          \
        a.push_back(TYPE(3.0));                          \
                                                         \
        EXPECT_EQ(a.empty(), false);                     \
    }

TEST_ARRAY_EMPTY(float, Float)
TEST_ARRAY_EMPTY(double, Double)
TEST_ARRAY_EMPTY(int, Int)
TEST_ARRAY_EMPTY(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_EMPTY(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_AT(TYPE, NAME)                        \
    TEST(ValueTest, Array##NAME##At) {                   \
        Array<Value<TYPE>> a;                            \
        a.push_back(TYPE(2.0));                          \
        a.push_back(TYPE(3.0));                          \
        a.push_back(TYPE(4.0));                          \
                                                         \
        EXPECT_EQ(a.at(0).dataX(), TYPE(2.0));           \
        EXPECT_EQ(a.at(1).dataX(), TYPE(3.0));           \
        EXPECT_EQ(a.at(2).dataX(), TYPE(4.0));           \
    }

TEST_ARRAY_AT(float, Float)
TEST_ARRAY_AT(double, Double)
TEST_ARRAY_AT(int, Int)
TEST_ARRAY_AT(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_AT(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_AT_EXCEPTION(TYPE, NAME)              \
    TEST(ValueTest, Array##NAME##AtException) {          \
        Array<Value<TYPE>> a;                            \
        a.push_back(TYPE(2.0));                          \
        a.push_back(TYPE(3.0));                          \
        a.push_back(TYPE(4.0));                          \
                                                         \
        EXPECT_THROW(a.at(3), std::out_of_range);        \
    }

TEST_ARRAY_AT_EXCEPTION(float, Float)
TEST_ARRAY_AT_EXCEPTION(double, Double)
TEST_ARRAY_AT_EXCEPTION(int, Int)
TEST_ARRAY_AT_EXCEPTION(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_AT_EXCEPTION(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D(TYPE, NAME)                         \
    TEST(ValueTest, Array##NAME##2D) {                    \
        Array<Array<Value<TYPE>>> a;                      \
        Array<Value<TYPE>> b;                             \
        b.push_back(TYPE(2.0));                           \
        b.push_back(TYPE(3.0));                           \
        b.push_back(TYPE(4.0));                           \
        a.push_back(b);                                   \
        a.push_back(b);                                   \
                                                          \
        EXPECT_EQ(a[0][0].dataX(), TYPE(2.0));            \
        EXPECT_EQ(a[0][1].dataX(), TYPE(3.0));            \
        EXPECT_EQ(a[0][2].dataX(), TYPE(4.0));            \
        EXPECT_EQ(a[1][0].dataX(), TYPE(2.0));            \
        EXPECT_EQ(a[1][1].dataX(), TYPE(3.0));            \
        EXPECT_EQ(a[1][2].dataX(), TYPE(4.0));            \
    }

TEST_ARRAY_2D(float, Float)
TEST_ARRAY_2D(double, Double)
TEST_ARRAY_2D(int, Int)
TEST_ARRAY_2D(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_AT(TYPE, NAME)                        \
    TEST(ValueTest, Array##NAME##2DAt) {                    \
        Array<Array<Value<TYPE>>> a;                        \
        Array<Value<TYPE>> b;                               \
        b.push_back(TYPE(2.0));                             \
        b.push_back(TYPE(3.0));                             \
        b.push_back(TYPE(4.0));                             \
        a.push_back(b);                                     \
        a.push_back(b);                                     \
                                                            \
        EXPECT_EQ(a.at(0).at(0).dataX(), TYPE(2.0));        \
        EXPECT_EQ(a.at(0).at(1).dataX(), TYPE(3.0));        \
        EXPECT_EQ(a.at(0).at(2).dataX(), TYPE(4.0));        \
        EXPECT_EQ(a.at(1).at(0).dataX(), TYPE(2.0));        \
        EXPECT_EQ(a.at(1).at(1).dataX(), TYPE(3.0));        \
        EXPECT_EQ(a.at(1).at(2).dataX(), TYPE(4.0));        \
    }

TEST_ARRAY_2D_AT(float, Float)
TEST_ARRAY_2D_AT(double, Double)
TEST_ARRAY_2D_AT(int, Int)
TEST_ARRAY_2D_AT(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_AT(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_AT_EXCEPTION(TYPE, NAME)              \
    TEST(ValueTest, Array##NAME##2DAtException) {           \
        Array<Array<Value<TYPE>>> a;                        \
        Array<Value<TYPE>> b;                               \
        b.push_back(TYPE(2.0));                             \
        b.push_back(TYPE(3.0));                             \
        b.push_back(TYPE(4.0));                             \
        a.push_back(b);                                     \
        a.push_back(b);                                     \
                                                            \
        EXPECT_THROW(a.at(2).at(0), std::out_of_range);     \
    }

TEST_ARRAY_2D_AT_EXCEPTION(float, Float)
TEST_ARRAY_2D_AT_EXCEPTION(double, Double)
TEST_ARRAY_2D_AT_EXCEPTION(int, Int)
TEST_ARRAY_2D_AT_EXCEPTION(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_AT_EXCEPTION(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_CLEAR(TYPE, NAME)                     \
    TEST(ValueTest, Array##NAME##2DClear) {                 \
        Array<Array<Value<TYPE>>> a;                        \
        Array<Value<TYPE>> b;                               \
        b.push_back(TYPE(2.0));                             \
        b.push_back(TYPE(3.0));                             \
        b.push_back(TYPE(4.0));                             \
        a.push_back(b);                                     \
        a.push_back(b);                                     \
                                                            \
        a.clear();                                          \
                                                            \
        EXPECT_EQ(a.size(), 0);                             \
    }

TEST_ARRAY_2D_CLEAR(float, Float)
TEST_ARRAY_2D_CLEAR(double, Double)
TEST_ARRAY_2D_CLEAR(int, Int)
TEST_ARRAY_2D_CLEAR(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_CLEAR(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_SIZE(TYPE, NAME)                      \
    TEST(ValueTest, Array##NAME##2DSize) {                  \
        Array<Array<Value<TYPE>>> a;                        \
        Array<Value<TYPE>> b;                               \
        b.push_back(TYPE(2.0));                             \
        b.push_back(TYPE(3.0));                             \
        b.push_back(TYPE(4.0));                             \
        a.push_back(b);                                     \
        a.push_back(b);                                     \
                                                            \
        EXPECT_EQ(a.size(), (2, 2));                        \
        EXPECT_EQ(a[0].size(), 3);                          \
        EXPECT_EQ(a[1].size(), 3);                          \
    }

TEST_ARRAY_2D_SIZE(float, Float)
TEST_ARRAY_2D_SIZE(double, Double)
TEST_ARRAY_2D_SIZE(int, Int)
TEST_ARRAY_2D_SIZE(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_SIZE(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_EMPTY(TYPE, NAME)                     \
    TEST(ValueTest, Array##NAME##2DEmpty) {                 \
        Array<Array<Value<TYPE>>> a;                        \
        EXPECT_EQ(a.empty(), true);                         \
                                                            \
        Array<Value<TYPE>> b;                               \
        b.push_back(TYPE(2.0));                             \
        b.push_back(TYPE(3.0));                             \
        a.push_back(b);                                     \
                                                            \
        EXPECT_EQ(a.empty(), false);                        \
    }

TEST_ARRAY_2D_EMPTY(float, Float)
TEST_ARRAY_2D_EMPTY(double, Double)
TEST_ARRAY_2D_EMPTY(int, Int)
TEST_ARRAY_2D_EMPTY(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_EMPTY(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_POPBACK(TYPE, NAME)                   \
    TEST(ValueTest, Array##NAME##2DPopBack) {               \
        Array<Array<Value<TYPE>>> a;                        \
        Array<Value<TYPE>> b;                               \
        b.push_back(TYPE(2.0));                             \
        b.push_back(TYPE(3.0));                             \
        b.push_back(TYPE(4.0));                             \
        a.push_back(b);                                     \
        a.push_back(b);                                     \
                                                            \
        a.pop_back();                                       \
                                                            \
        EXPECT_EQ(a.size(), 1);                             \
    }

TEST_ARRAY_2D_POPBACK(float, Float)
TEST_ARRAY_2D_POPBACK(double, Double)
TEST_ARRAY_2D_POPBACK(int, Int)
TEST_ARRAY_2D_POPBACK(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_POPBACK(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_ADD(TYPE, NAME)                           \
    TEST(ValueTest, Array##NAME##Add) {                      \
        Array<Value<TYPE>> a;                                \
        a.push_back(TYPE(2.0));                              \
        a.push_back(TYPE(3.0));                              \
        a.push_back(TYPE(4.0));                              \
                                                             \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(5.0));                              \
        b.push_back(TYPE(6.0));                              \
        b.push_back(TYPE(7.0));                              \
                                                             \
        Array<Value<TYPE>> c = a + b;                        \
                                                             \
        EXPECT_EQ(c[0].dataX(), TYPE(7.0));                  \
        EXPECT_EQ(c[1].dataX(), TYPE(9.0));                  \
        EXPECT_EQ(c[2].dataX(), TYPE(11.0));                 \
                                                             \
        Array<Value<TYPE>> d = ptMgrad::add(a, b);           \
                                                             \
        EXPECT_EQ(d[0].dataX(), TYPE(7.0));                  \
        EXPECT_EQ(d[1].dataX(), TYPE(9.0));                  \
        EXPECT_EQ(d[2].dataX(), TYPE(11.0));                 \
    }

TEST_ARRAY_ADD(float, Float)
TEST_ARRAY_ADD(double, Double)
TEST_ARRAY_ADD(int, Int)
TEST_ARRAY_ADD(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_ADD(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_ADD(TYPE, NAME)                        \
    TEST(ValueTest, Array##NAME##2DAdd) {                    \
        Array<Array<Value<TYPE>>> a;                         \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(2.0));                              \
        b.push_back(TYPE(3.0));                              \
        b.push_back(TYPE(4.0));                              \
        a.push_back(b);                                      \
        a.push_back(b);                                      \
                                                             \
        Array<Array<Value<TYPE>>> c;                         \
        Array<Value<TYPE>> d;                                \
        d.push_back(TYPE(5.0));                              \
        d.push_back(TYPE(6.0));                              \
        d.push_back(TYPE(7.0));                              \
        c.push_back(d);                                      \
        c.push_back(d);                                      \
                                                             \
        Array<Array<Value<TYPE>>> e = a + c;                 \
                                                             \
        EXPECT_EQ(e[0][0].dataX(), TYPE(7.0));               \
        EXPECT_EQ(e[0][1].dataX(), TYPE(9.0));               \
        EXPECT_EQ(e[0][2].dataX(), TYPE(11.0));              \
        EXPECT_EQ(e[1][0].dataX(), TYPE(7.0));               \
        EXPECT_EQ(e[1][1].dataX(), TYPE(9.0));               \
        EXPECT_EQ(e[1][2].dataX(), TYPE(11.0));              \
    }

TEST_ARRAY_2D_ADD(float, Float)
TEST_ARRAY_2D_ADD(double, Double)
TEST_ARRAY_2D_ADD(int, Int)
TEST_ARRAY_2D_ADD(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_ADD(ptMgrad::complex<double>, ComplexDouble)

/*
#define TEST_ARRAY_2D_ADD_FUNC(TYPE, NAME)                                \
    TEST(ValueTest, Array##NAME##AddFunc) {                               \
        Array<Array<Value<TYPE>>> a;                                      \
        a.push_back(TYPE(2.0));                                           \
        a.push_back(TYPE(3.0));                                           \
        a.push_back(TYPE(4.0));                                           \
        b.push_back(a);                                                   \
        b.push_back(a);                                                   \
                                                                          \
        Array<Array<Value<TYPE>>> c = ptMgrad::add(a, TYPE(1.0));         \
                                                                          \
        EXPECT_EQ(c[0][0].dataX(), TYPE(3.0));                            \
        EXPECT_EQ(c[0][1].dataX(), TYPE(4.0));                            \
        EXPECT_EQ(c[0][2].dataX(), TYPE(5.0));                            \
        EXPECT_EQ(c[1][0].dataX(), TYPE(3.0));                            \
        EXPECT_EQ(c[1][1].dataX(), TYPE(4.0));                            \
        EXPECT_EQ(c[1][2].dataX(), TYPE(5.0));                            \
    }

TEST_ARRAY_2D_ADD_FUNC(float, Float)
TEST_ARRAY_2D_ADD_FUNC(double, Double)
TEST_ARRAY_2D_ADD_FUNC(int, Int)
TEST_ARRAY_2D_ADD_FUNC(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_ADD_FUNC(ptMgrad::complex<double>, ComplexDouble)
*/

#define TEST_ARRAY_SUB(TYPE, NAME)                           \
    TEST(ValueTest, Array##NAME##Sub) {                      \
        Array<Value<TYPE>> a;                                \
        a.push_back(TYPE(2.0));                              \
        a.push_back(TYPE(3.0));                              \
        a.push_back(TYPE(4.0));                              \
                                                             \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(-1.0));                             \
        b.push_back(TYPE(-2.0));                             \
        b.push_back(TYPE(-3.0));                             \
                                                             \
        Array<Value<TYPE>> c = a - b;                        \
                                                             \
        EXPECT_EQ(c[0].dataX(), TYPE(3.0));                  \
        EXPECT_EQ(c[1].dataX(), TYPE(5.0));                  \
        EXPECT_EQ(c[2].dataX(), TYPE(7.0));                  \
                                                             \
        Array<Value<TYPE>> d = ptMgrad::sub(a, b);           \
                                                             \
        EXPECT_EQ(d[0].dataX(), TYPE(3.0));                  \
        EXPECT_EQ(d[1].dataX(), TYPE(5.0));                  \
        EXPECT_EQ(d[2].dataX(), TYPE(7.0));                  \
    }

TEST_ARRAY_SUB(float, Float)
TEST_ARRAY_SUB(double, Double)
TEST_ARRAY_SUB(int, Int)
TEST_ARRAY_SUB(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_SUB(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_SUB(TYPE, NAME)                        \
    TEST(ValueTest, Array##NAME##2DSub) {                    \
        Array<Array<Value<TYPE>>> a;                         \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(2.0));                              \
        b.push_back(TYPE(3.0));                              \
        b.push_back(TYPE(4.0));                              \
        a.push_back(b);                                      \
        a.push_back(b);                                      \
                                                             \
        Array<Array<Value<TYPE>>> c;                         \
        Array<Value<TYPE>> d;                                \
        d.push_back(TYPE(1.0));                              \
        d.push_back(TYPE(1.0));                              \
        d.push_back(TYPE(1.0));                              \
        c.push_back(d);                                      \
        c.push_back(d);                                      \
                                                             \
        Array<Array<Value<TYPE>>> e = a - c;                 \
                                                             \
        EXPECT_EQ(e[0][0].dataX(), TYPE(1.0));               \
        EXPECT_EQ(e[0][1].dataX(), TYPE(2.0));               \
        EXPECT_EQ(e[0][2].dataX(), TYPE(3.0));               \
        EXPECT_EQ(e[1][0].dataX(), TYPE(1.0));               \
        EXPECT_EQ(e[1][1].dataX(), TYPE(2.0));               \
        EXPECT_EQ(e[1][2].dataX(), TYPE(3.0));               \
    }

TEST_ARRAY_2D_SUB(float, Float)
TEST_ARRAY_2D_SUB(double, Double)
TEST_ARRAY_2D_SUB(int, Int)
TEST_ARRAY_2D_SUB(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_SUB(ptMgrad::complex<double>, ComplexDouble)



#define TEST_ARRAY_SUB_SCALAR_FUNC(TYPE, NAME)                    \
    TEST(ValueTest, Array##NAME##SubFunc) {                       \
        Array<Value<TYPE>> a;                                     \
        a.push_back(TYPE(2.0));                                   \
        a.push_back(TYPE(3.0));                                   \
        a.push_back(TYPE(4.0));                                   \
                                                                  \
        Array<Value<TYPE>> b = ptMgrad::sub(a, TYPE(1.0));        \
                                                                  \
        EXPECT_EQ(b[0].dataX(), TYPE(1.0));                       \
        EXPECT_EQ(b[1].dataX(), TYPE(2.0));                       \
        EXPECT_EQ(b[2].dataX(), TYPE(3.0));                       \
                                                                  \
        Array<Value<TYPE>> c = a - TYPE(1.0);                     \
                                                                  \
        EXPECT_EQ(c[0].dataX(), TYPE(1.0));                       \
        EXPECT_EQ(c[1].dataX(), TYPE(2.0));                       \
        EXPECT_EQ(c[2].dataX(), TYPE(3.0));                       \
    }

TEST_ARRAY_SUB_SCALAR_FUNC(float, Float)
TEST_ARRAY_SUB_SCALAR_FUNC(double, Double)
TEST_ARRAY_SUB_SCALAR_FUNC(int, Int)
TEST_ARRAY_SUB_SCALAR_FUNC(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_SUB_SCALAR_FUNC(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_MUL(TYPE, NAME)                           \
    TEST(ValueTest, Array##NAME##Mul) {                      \
        Array<Value<TYPE>> a;                                \
        a.push_back(TYPE(2.0));                              \
        a.push_back(TYPE(3.0));                              \
        a.push_back(TYPE(4.0));                              \
                                                             \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(5.0));                              \
        b.push_back(TYPE(6.0));                              \
        b.push_back(TYPE(7.0));                              \
                                                             \
        Array<Value<TYPE>> c = a * b;                        \
                                                             \
        EXPECT_EQ(c[0].dataX(), TYPE(10.0));                 \
        EXPECT_EQ(c[1].dataX(), TYPE(18.0));                 \
        EXPECT_EQ(c[2].dataX(), TYPE(28.0));                 \
                                                             \
        Array<Value<TYPE>> d = ptMgrad::mul(a, b);           \
                                                             \
        EXPECT_EQ(d[0].dataX(), TYPE(10.0));                 \
        EXPECT_EQ(d[1].dataX(), TYPE(18.0));                 \
        EXPECT_EQ(d[2].dataX(), TYPE(28.0));                 \
    }

TEST_ARRAY_MUL(float, Float)
TEST_ARRAY_MUL(double, Double)
TEST_ARRAY_MUL(int, Int)
TEST_ARRAY_MUL(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_MUL(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_MUL(TYPE, NAME)                        \
    TEST(ValueTest, Array##NAME##2DMul) {                    \
        Array<Array<Value<TYPE>>> a;                         \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(2.0));                              \
        b.push_back(TYPE(3.0));                              \
        b.push_back(TYPE(4.0));                              \
        a.push_back(b);                                      \
        a.push_back(b);                                      \
                                                             \
        Array<Array<Value<TYPE>>> c;                         \
        Array<Value<TYPE>> d;                                \
        d.push_back(TYPE(5.0));                              \
        d.push_back(TYPE(6.0));                              \
        d.push_back(TYPE(7.0));                              \
        c.push_back(d);                                      \
        c.push_back(d);                                      \
                                                             \
        Array<Array<Value<TYPE>>> e = a * c;                 \
                                                             \
        EXPECT_EQ(e[0][0].dataX(), TYPE(10.0));              \
        EXPECT_EQ(e[0][1].dataX(), TYPE(18.0));              \
        EXPECT_EQ(e[0][2].dataX(), TYPE(28.0));              \
        EXPECT_EQ(e[1][0].dataX(), TYPE(10.0));              \
        EXPECT_EQ(e[1][1].dataX(), TYPE(18.0));              \
        EXPECT_EQ(e[1][2].dataX(), TYPE(28.0));              \
    }

TEST_ARRAY_2D_MUL(float, Float)
TEST_ARRAY_2D_MUL(double, Double)
TEST_ARRAY_2D_MUL(int, Int)
TEST_ARRAY_2D_MUL(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_MUL(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_DIV(TYPE, NAME)                           \
    TEST(ValueTest, Array##NAME##Div) {                      \
        Array<Value<TYPE>> a;                                \
        a.push_back(TYPE(2.0));                              \
        a.push_back(TYPE(3.0));                              \
        a.push_back(TYPE(4.0));                              \
                                                             \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(5.0));                              \
        b.push_back(TYPE(2.0));                              \
        b.push_back(TYPE(2.0));                              \
                                                             \
        Array<Value<TYPE>> c = a / b;                        \
                                                             \
        EXPECT_EQ(c[0].dataX(), TYPE(2.0 / 5.0));            \
        EXPECT_EQ(c[1].dataX(), TYPE(3.0 / 2.0));            \
        EXPECT_EQ(c[2].dataX(), TYPE(4.0 / 2.0));            \
                                                             \
        Array<Value<TYPE>> d = ptMgrad::div(a, b);           \
                                                             \
        EXPECT_EQ(d[0].dataX(), TYPE(2.0 / 5.0));            \
        EXPECT_EQ(d[1].dataX(), TYPE(3.0 / 2.0));            \
        EXPECT_EQ(d[2].dataX(), TYPE(4.0 / 2.0));            \
    }

TEST_ARRAY_DIV(float, Float)
TEST_ARRAY_DIV(double, Double)
//TEST_ARRAY_DIV(int, Int)
//TEST_ARRAY_DIV(ptMgrad::complex<float>, ComplexFloat)
//TEST_ARRAY_DIV(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_DIV(TYPE, NAME)                        \
    TEST(ValueTest, Array##NAME##2DDiv) {                    \
        Array<Array<Value<TYPE>>> a;                         \
        Array<Value<TYPE>> b;                                \
        b.push_back(TYPE(2.0));                              \
        b.push_back(TYPE(3.0));                              \
        b.push_back(TYPE(4.0));                              \
        a.push_back(b);                                      \
        a.push_back(b);                                      \
                                                             \
        Array<Array<Value<TYPE>>> c;                         \
        Array<Value<TYPE>> d;                                \
        d.push_back(TYPE(5.0));                              \
        d.push_back(TYPE(2.0));                              \
        d.push_back(TYPE(2.0));                              \
        c.push_back(d);                                      \
        c.push_back(d);                                      \
                                                             \
        Array<Array<Value<TYPE>>> e = a / c;                 \
                                                             \
        EXPECT_EQ(e[0][0].dataX(), TYPE(2.0 / 5.0));         \
        EXPECT_EQ(e[0][1].dataX(), TYPE(3.0 / 2.0));         \
        EXPECT_EQ(e[0][2].dataX(), TYPE(4.0 / 2.0));         \
        EXPECT_EQ(e[1][0].dataX(), TYPE(2.0 / 5.0));         \
        EXPECT_EQ(e[1][1].dataX(), TYPE(3.0 / 2.0));         \
        EXPECT_EQ(e[1][2].dataX(), TYPE(4.0 / 2.0));         \
    }

TEST_ARRAY_2D_DIV(float, Float)
TEST_ARRAY_2D_DIV(double, Double)
//TEST_ARRAY_2D_DIV(int, Int)
//TEST_ARRAY_2D_DIV(ptMgrad::complex<float>, ComplexFloat)
//TEST_ARRAY_2D_DIV(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_ADD_SCALAR_FUNC(TYPE, NAME)                    \
    TEST(ValueTest, Array##NAME##AddFunc) {                       \
        Array<Value<TYPE>> a;                                     \
        a.push_back(TYPE(2.0));                                   \
        a.push_back(TYPE(3.0));                                   \
        a.push_back(TYPE(4.0));                                   \
                                                                  \
        Array<Value<TYPE>> b = ptMgrad::add(a, TYPE(1.0));        \
                                                                  \
        EXPECT_EQ(b[0].dataX(), TYPE(3.0));                       \
        EXPECT_EQ(b[1].dataX(), TYPE(4.0));                       \
        EXPECT_EQ(b[2].dataX(), TYPE(5.0));                       \
                                                                  \
        Array<Value<TYPE>> c = a + TYPE(1.0);                     \
                                                                  \
        EXPECT_EQ(c[0].dataX(), TYPE(3.0));                       \
        EXPECT_EQ(c[1].dataX(), TYPE(4.0));                       \
        EXPECT_EQ(c[2].dataX(), TYPE(5.0));                       \
    }

TEST_ARRAY_ADD_SCALAR_FUNC(float, Float)
TEST_ARRAY_ADD_SCALAR_FUNC(double, Double)
TEST_ARRAY_ADD_SCALAR_FUNC(int, Int)
TEST_ARRAY_ADD_SCALAR_FUNC(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_ADD_SCALAR_FUNC(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_2D_ADD_FUNC2(TYPE, NAME)                              \
    TEST(ValueTest, Array##NAME##AddFunc2) {                             \
        Array<Value<TYPE>> a;                                            \
        a.push_back(TYPE(2.0));                                          \
        a.push_back(TYPE(3.0));                                          \
        a.push_back(TYPE(4.0));                                          \
                                                                         \
        Array<Array<Value<TYPE>>> b;                                     \
        b.push_back(a);                                                  \
        b.push_back(a);                                                  \
                                                                         \
        Array<Array<Value<TYPE>>> c = ptMgrad::add(b, TYPE(1.0));        \
                                                                         \
        EXPECT_EQ(c[0][0].dataX(), TYPE(3.0));                           \
        EXPECT_EQ(c[0][1].dataX(), TYPE(4.0));                           \
        EXPECT_EQ(c[0][2].dataX(), TYPE(5.0));                           \
        EXPECT_EQ(c[1][0].dataX(), TYPE(3.0));                           \
        EXPECT_EQ(c[1][1].dataX(), TYPE(4.0));                           \
        EXPECT_EQ(c[1][2].dataX(), TYPE(5.0));                           \
                                                                         \
        Array<Array<Value<TYPE>>> d = b + TYPE(1.0);                     \
                                                                         \
		EXPECT_EQ(d[0][0].dataX(), TYPE(3.0));                           \
        EXPECT_EQ(d[0][1].dataX(), TYPE(4.0));                           \
        EXPECT_EQ(d[0][2].dataX(), TYPE(5.0));                           \
        EXPECT_EQ(d[1][0].dataX(), TYPE(3.0));                           \
        EXPECT_EQ(d[1][1].dataX(), TYPE(4.0));                           \
        EXPECT_EQ(d[1][2].dataX(), TYPE(5.0));                           \
    }

TEST_ARRAY_2D_ADD_FUNC2(float, Float)
TEST_ARRAY_2D_ADD_FUNC2(double, Double)
TEST_ARRAY_2D_ADD_FUNC2(int, Int)
TEST_ARRAY_2D_ADD_FUNC2(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_2D_ADD_FUNC2(ptMgrad::complex<double>, ComplexDouble)

/*
#define TEST_ARRAY_SUB_SCALAR_FUNC(TYPE, NAME)                    \
    TEST(ValueTest, Array##NAME##SubFunc) {                       \
        Array<Value<TYPE>> a;                                     \
        a.push_back(TYPE(2.0));                                   \
        a.push_back(TYPE(3.0));                                   \
        a.push_back(TYPE(4.0));                                   \
                                                                  \
        Array<Value<TYPE>> b = ptMgrad::sub(a, TYPE(1.0));        \
                                                                  \
        EXPECT_EQ(b[0].dataX(), TYPE(1.0));                       \
        EXPECT_EQ(b[1].dataX(), TYPE(2.0));                       \
        EXPECT_EQ(b[2].dataX(), TYPE(3.0));                       \
                                                                  \
        Array<Value<TYPE>> c = a - TYPE(1.0);                     \
                                                                  \
        EXPECT_EQ(c[0].dataX(), TYPE(1.0));                       \
        EXPECT_EQ(c[1].dataX(), TYPE(2.0));                       \
        EXPECT_EQ(c[2].dataX(), TYPE(3.0));                       \
    }

TEST_ARRAY_SUB_SCALAR_FUNC(float, Float)
TEST_ARRAY_SUB_SCALAR_FUNC(double, Double)
TEST_ARRAY_SUB_SCALAR_FUNC(int, Int)
TEST_ARRAY_SUB_SCALAR_FUNC(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_SUB_SCALAR_FUNC(ptMgrad::complex<double>, ComplexDouble)
*/

#define TEST_ARRAY_MUL_SCALAR_FUNC(TYPE, NAME)                    \
    TEST(ValueTest, Array##NAME##MulFunc) {                       \
        Array<Value<TYPE>> a;                                     \
        a.push_back(TYPE(2.0));                                   \
        a.push_back(TYPE(3.0));                                   \
        a.push_back(TYPE(4.0));                                   \
                                                                  \
        Array<Value<TYPE>> b = ptMgrad::mul(a, TYPE(2.0));        \
                                                                  \
        EXPECT_EQ(b[0].dataX(), TYPE(4.0));                       \
        EXPECT_EQ(b[1].dataX(), TYPE(6.0));                       \
        EXPECT_EQ(b[2].dataX(), TYPE(8.0));                       \
                                                                  \
        Array<Value<TYPE>> c = a * TYPE(2.0);                     \
                                                                  \
        EXPECT_EQ(c[0].dataX(), TYPE(4.0));                       \
        EXPECT_EQ(c[1].dataX(), TYPE(6.0));                       \
        EXPECT_EQ(c[2].dataX(), TYPE(8.0));                       \
    }

TEST_ARRAY_MUL_SCALAR_FUNC(float, Float)
TEST_ARRAY_MUL_SCALAR_FUNC(double, Double)
TEST_ARRAY_MUL_SCALAR_FUNC(int, Int)
TEST_ARRAY_MUL_SCALAR_FUNC(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_MUL_SCALAR_FUNC(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_DIV_SCALAR_FUNC(TYPE, NAME)                    \
    TEST(ValueTest, Array##NAME##DivFunc) {                       \
        Array<Value<TYPE>> a;                                     \
        a.push_back(TYPE(2.0));                                   \
        a.push_back(TYPE(3.0));                                   \
        a.push_back(TYPE(4.0));                                   \
                                                                  \
        Array<Value<TYPE>> b = ptMgrad::div(a, TYPE(2.0));        \
                                                                  \
        EXPECT_EQ(b[0].dataX(), TYPE(1.0));                       \
        EXPECT_EQ(b[1].dataX(), TYPE(1.5));                       \
        EXPECT_EQ(b[2].dataX(), TYPE(2.0));                       \
                                                                  \
        Array<Value<TYPE>> c = a / TYPE(2.0);                     \
                                                                  \
        EXPECT_EQ(c[0].dataX(), TYPE(1.0));                       \
        EXPECT_EQ(c[1].dataX(), TYPE(1.5));                       \
        EXPECT_EQ(c[2].dataX(), TYPE(2.0));                       \
    }

TEST_ARRAY_DIV_SCALAR_FUNC(float, Float)
TEST_ARRAY_DIV_SCALAR_FUNC(double, Double)
TEST_ARRAY_DIV_SCALAR_FUNC(int, Int)
TEST_ARRAY_DIV_SCALAR_FUNC(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_DIV_SCALAR_FUNC(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_NEG_FUNC(TYPE, NAME)                      \
    TEST(ValueTest, Array##NAME##NegFunc) {                  \
        Array<Value<TYPE>> a;                                \
        a.push_back(TYPE(2.0));                              \
        a.push_back(TYPE(3.0));                              \
        a.push_back(TYPE(4.0));                              \
                                                             \
        Array<Value<TYPE>> b;                                \
                                                             \
        for (int i = 0; i < a.size(); i++) {                 \
            b.push_back(ptMgrad::neg(a[i]));                 \
        }                                                    \
                                                             \
        EXPECT_EQ(b[0].dataX(), TYPE(-2.0));                 \
        EXPECT_EQ(b[1].dataX(), TYPE(-3.0));                 \
        EXPECT_EQ(b[2].dataX(), TYPE(-4.0));                 \
    }

TEST_ARRAY_NEG_FUNC(float, Float)
TEST_ARRAY_NEG_FUNC(double, Double)
TEST_ARRAY_NEG_FUNC(int, Int)
TEST_ARRAY_NEG_FUNC(ptMgrad::complex<float>, ComplexFloat)
TEST_ARRAY_NEG_FUNC(ptMgrad::complex<double>, ComplexDouble)


#define TEST_ARRAY_DIFF_TYPES_ADD(TYPE1, TYPE2, NAME1, NAME2)                 \
    TEST(ValueTest, ArrayAdd##NAME1##NAME2##Types) {                          \
        Array<Value<TYPE1>> a = {TYPE1(2.0), TYPE1(3.0), TYPE1(4.0)};         \
                                                                              \
        Array<Value<TYPE2>> b = {TYPE2(5.0), TYPE2(6.0), TYPE2(7.0)};         \
                                                                              \
        using ResultType = typename std::common_type_t<TYPE1, TYPE2>;         \
                                                                              \
        Array<Value<ResultType>> c = a + b;                                   \
                                                                              \
        EXPECT_EQ(c[0].dataX(), ResultType(7.0));                             \
        EXPECT_EQ(c[1].dataX(), ResultType(9.0));                             \
        EXPECT_EQ(c[2].dataX(), ResultType(11.0));                            \
                                                                              \
        Array<Value<ResultType>> d = ptMgrad::add(a, b);                      \
                                                                              \
        EXPECT_EQ(d[0].dataX(), ResultType(7.0));                             \
        EXPECT_EQ(d[1].dataX(), ResultType(9.0));                             \
        EXPECT_EQ(d[2].dataX(), ResultType(11.0));                            \
    }

//TEST_ARRAY_DIFF_TYPES_ADD(float, double, Float, Double)
TEST_ARRAY_DIFF_TYPES_ADD(double, float, Double, Float)
TEST_ARRAY_DIFF_TYPES_ADD(float, int, Float, Int)
//TEST_ARRAY_DIFF_TYPES_ADD(int, float, Int, Float)


#define TEST_ARRAY_DIFF_TYPES_ADD_SCALAR_FUNC(TYPE1, TYPE2, NAME1, NAME2)            \
    TEST(ValueTest, ArrayAddScalar##NAME1##NAME2##Types) {                           \
        Array<Value<TYPE1>> a = {TYPE1(2.0), TYPE1(3.0), TYPE1(4.0)};                \
                                                                                     \
        using ResultType = typename std::common_type_t<TYPE1, TYPE2>;                \
                                                                                     \
        Array<Value<ResultType>> b = ptMgrad::add(a, TYPE2(1.0));                    \
                                                                                     \
		EXPECT_EQ(b[0].dataX(), ResultType(3.0));                                    \
        EXPECT_EQ(b[1].dataX(), ResultType(4.0));                                    \
        EXPECT_EQ(b[2].dataX(), ResultType(5.0));                                    \
                                                                                     \
        Array<Value<ResultType>> c = a + TYPE2(1.0);                                 \
                                                                                     \
		EXPECT_EQ(c[0].dataX(), ResultType(3.0));                                    \
        EXPECT_EQ(c[1].dataX(), ResultType(4.0));                                    \
        EXPECT_EQ(c[2].dataX(), ResultType(5.0));                                    \
    }

TEST_ARRAY_DIFF_TYPES_ADD_SCALAR_FUNC(double, float, Float, Double)
TEST_ARRAY_DIFF_TYPES_ADD_SCALAR_FUNC(float, int, Double, Float)


#define TEST_ARRAY_DIFF_TYPES_SUB(TYPE1, TYPE2, NAME1, NAME2)                 \
    TEST(ValueTest, ArraySub##NAME1##NAME2##Types) {                          \
        Array<Value<TYPE1>> a = {TYPE1(2.0), TYPE1(3.0), TYPE1(4.0)};         \
                                                                              \
        Array<Value<TYPE2>> b = {TYPE2(5.0), TYPE2(6.0), TYPE2(7.0)};         \
                                                                              \
        using ResultType = typename std::common_type_t<TYPE1, TYPE2>;         \
                                                                              \
        Array<Value<ResultType>> c = a - b;                                   \
                                                                              \
        EXPECT_EQ(c[0].dataX(), ResultType(-3.0));                            \
        EXPECT_EQ(c[1].dataX(), ResultType(-3.0));                            \
        EXPECT_EQ(c[2].dataX(), ResultType(-3.0));                            \
                                                                              \
        Array<Value<ResultType>> d = ptMgrad::sub(a, b);                      \
                                                                              \
        EXPECT_EQ(d[0].dataX(), ResultType(-3.0));                            \
        EXPECT_EQ(d[1].dataX(), ResultType(-3.0));                            \
        EXPECT_EQ(d[2].dataX(), ResultType(-3.0));                            \
    }

TEST_ARRAY_DIFF_TYPES_SUB(double, float, Float, Double)
//TEST_ARRAY_DIFF_TYPES_SUB(float, int, Double, Float)


#define TEST_ARRAY_DIFF_TYPES_MUL(TYPE1, TYPE2, NAME1, NAME2)                 \
    TEST(ValueTest, ArrayMul##NAME1##NAME2##Types) {                          \
        Array<Value<TYPE1>> a = {TYPE1(2.0), TYPE1(3.0), TYPE1(4.0)};         \
                                                                              \
        Array<Value<TYPE2>> b = {TYPE2(5.0), TYPE2(6.0), TYPE2(7.0)};         \
                                                                              \
        using ResultType = typename std::common_type_t<TYPE1, TYPE2>;         \
                                                                              \
        Array<Value<ResultType>> c = a * b;                                   \
                                                                              \
        EXPECT_EQ(c[0].dataX(), ResultType(10.0));                            \
        EXPECT_EQ(c[1].dataX(), ResultType(18.0));                            \
        EXPECT_EQ(c[2].dataX(), ResultType(28.0));                            \
                                                                              \
        Array<Value<ResultType>> d = ptMgrad::mul(a, b);                      \
                                                                              \
        EXPECT_EQ(d[0].dataX(), ResultType(10.0));                            \
        EXPECT_EQ(d[1].dataX(), ResultType(18.0));                            \
        EXPECT_EQ(d[2].dataX(), ResultType(28.0));                            \
    }

TEST_ARRAY_DIFF_TYPES_MUL(double, float, Float, Double)
TEST_ARRAY_DIFF_TYPES_MUL(float, int, Double, Float)
//TEST_ARRAY_DIFF_TYPES_MUL(int, float, Int, Float)


#define TEST_ARRAY_DIFF_TYPES_MUL_SCALAR_FUNC(TYPE1, TYPE2, NAME1, NAME2)            \
    TEST(ValueTest, ArrayMulScalar##NAME1##NAME2##Types) {                           \
        Array<Value<TYPE1>> a = {TYPE1(2.0), TYPE1(3.0), TYPE1(4.0)};                \
                                                                                     \
        using ResultType = typename std::common_type_t<TYPE1, TYPE2>;                \
                                                                                     \
        Array<Value<ResultType>> b = ptMgrad::mul(a, TYPE2(2.0));                    \
                                                                                     \
		EXPECT_EQ(b[0].dataX(), ResultType(4.0));                                    \
        EXPECT_EQ(b[1].dataX(), ResultType(6.0));                                    \
        EXPECT_EQ(b[2].dataX(), ResultType(8.0));                                    \
                                                                                     \
		Array<Value<ResultType>> c = a * TYPE2(2.0);                                 \
                                                                                     \
		EXPECT_EQ(c[0].dataX(), ResultType(4.0));                                    \
        EXPECT_EQ(c[1].dataX(), ResultType(6.0));                                    \
        EXPECT_EQ(c[2].dataX(), ResultType(8.0));                                    \
    }

TEST_ARRAY_DIFF_TYPES_MUL_SCALAR_FUNC(double, float, Float, Double)
TEST_ARRAY_DIFF_TYPES_MUL_SCALAR_FUNC(float, int, Double, Float)


#define TEST_ARRAY_DIFF_TYPES_DIV(TYPE1, TYPE2, NAME1, NAME2)                 \
    TEST(ValueTest, ArrayDiv##NAME1##NAME2##Types) {                          \
        Array<Value<TYPE1>> a = {TYPE1(2.0), TYPE1(3.0), TYPE1(4.0)};         \
                                                                              \
        Array<Value<TYPE2>> b = {TYPE2(5.0), TYPE2(6.0), TYPE2(2.0)};         \
                                                                              \
        using ResultType = typename std::common_type_t<TYPE1, TYPE2>;         \
                                                                              \
        Array<Value<ResultType>> c = a / b;                                   \
                                                                              \
        EXPECT_EQ(c[0].dataX(), ResultType(0.4));                             \
        EXPECT_EQ(c[1].dataX(), ResultType(0.5));                             \
        EXPECT_EQ(c[2].dataX(), ResultType(2.0));                             \
                                                                              \
        Array<Value<ResultType>> d = ptMgrad::div(a, b);                      \
                                                                              \
        EXPECT_EQ(d[0].dataX(), ResultType(0.4));                             \
        EXPECT_EQ(d[1].dataX(), ResultType(0.5));                             \
        EXPECT_EQ(d[2].dataX(), ResultType(2.0));                             \
    }

TEST_ARRAY_DIFF_TYPES_DIV(double, float, Float, Double)
TEST_ARRAY_DIFF_TYPES_DIV(float, int, Double, Float)


#define TEST_ARRAY_DIFF_TYPES_DIV_SCALAR_FUNC(TYPE1, TYPE2, NAME1, NAME2)            \
    TEST(ValueTest, ArrayDivScalar##NAME1##NAME2##Types) {                           \
        Array<Value<TYPE1>> a = {TYPE1(2.0), TYPE1(3.0), TYPE1(4.0)};                \
                                                                                     \
        using ResultType = typename std::common_type_t<TYPE1, TYPE2>;                \
                                                                                     \
        Array<Value<ResultType>> b = ptMgrad::div(a, TYPE2(2.0));                    \
                                                                                     \
		EXPECT_EQ(b[0].dataX(), ResultType(1.0));                                    \
        EXPECT_EQ(b[1].dataX(), ResultType(1.5));                                    \
        EXPECT_EQ(b[2].dataX(), ResultType(2.0));                                    \
                                                                                     \
		Array<Value<ResultType>> c = a / TYPE2(2.0);                                 \
                                                                                     \
		EXPECT_EQ(c[0].dataX(), ResultType(1.0));                                    \
        EXPECT_EQ(c[1].dataX(), ResultType(1.5));                                    \
        EXPECT_EQ(c[2].dataX(), ResultType(2.0));                                    \
    }

TEST_ARRAY_DIFF_TYPES_DIV_SCALAR_FUNC(double, float, Float, Double)
TEST_ARRAY_DIFF_TYPES_DIV_SCALAR_FUNC(float, int, Double, Float)
