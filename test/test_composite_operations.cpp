#include <iostream>

#include <gtest/gtest.h>


#include "../src/engine.h"
#include "../src/Array.h"
#include "../src/complex.h"


using namespace ptMgrad;


#define TEST_COMPOSITE_OPS(TYPE, NAME)                       \
    TEST(ValueTest, Composite##NAME##Operations) {           \
        Value<TYPE> a = 2.0;                                 \
        Value<TYPE> b = 3.0;                                 \
        Value<TYPE> c = 4.0;                                 \
        Value<TYPE> d = 5.0;                                 \
                                                             \
        Value<TYPE> e = a + b * c - d;                       \
                                                             \
        EXPECT_EQ(e.dataX(), TYPE(9.0));                     \
    }

TEST_COMPOSITE_OPS(float, Float)
TEST_COMPOSITE_OPS(double, Double)
TEST_COMPOSITE_OPS(int, Int)
TEST_COMPOSITE_OPS(ptMgrad::complex<float>, ComplexFloat)
TEST_COMPOSITE_OPS(ptMgrad::complex<double>, ComplexDouble)

/*
#define TEST_COMPOSITE_OPS2(TYPE, NAME)                        \
    TEST(ValueTest, Composite##NAME##Operations2) {            \
        Value<TYPE> a = 2.0;                                   \
        Value<TYPE> b = 3.0;                                   \
        Value<TYPE> c = 4.0;                                   \
        Value<TYPE> d = 5.0;                                   \
                                                               \
        Value<TYPE> e = 5 * a + 2 * b / c - d;                 \
                                                               \
        EXPECT_EQ(e.dataX(), TYPE(6.5));                       \
    }

TEST_COMPOSITE_OPS2(float, Float)
TEST_COMPOSITE_OPS2(double, Double)
TEST_COMPOSITE_OPS2(int, Int)
TEST_COMPOSITE_OPS2(ptMgrad::complex<float>, ComplexFloat)
TEST_COMPOSITE_OPS2(ptMgrad::complex<double>, ComplexDouble)


#define TEST_COMPOSITE_OPS3(TYPE, NAME)                        \
    TEST(ValueTest, Composite##NAME##Operations3) {            \
        Value<TYPE> a = 2.0;                                   \
        Value<TYPE> b = 3.0;                                   \
        Value<TYPE> c = 4.0;                                   \
        Value<TYPE> d = 5.0;                                   \
                                                               \
        Value<TYPE> e = 2 * a / (1 / d);                       \
                                                               \
        EXPECT_EQ(e.dataX(), TYPE(20.0));                      \
    }

TEST_COMPOSITE_OPS3(float, Float)
TEST_COMPOSITE_OPS3(double, Double)
TEST_COMPOSITE_OPS3(int, Int)
TEST_COMPOSITE_OPS3(ptMgrad::complex<float>, ComplexFloat)
TEST_COMPOSITE_OPS3(ptMgrad::complex<double>, ComplexDouble)
*/

#define TEST_COMPOSITE_OPS4(TYPE, NAME)                        \
	TEST(ValueTest, Composite##NAME##Operations4) {            \
		Value<TYPE> a = 2.0;                                   \
		Value<TYPE> b = 3.0;                                   \
		Value<TYPE> c = 4.0;                                   \
		Value<TYPE> d = 5.0;                                   \
                                                               \
		Value<TYPE> e = ptMgrad::pow(a, b) + 1;                \
                                                               \
		EXPECT_EQ(e.dataX(), TYPE(9.0));                       \
	}

TEST_COMPOSITE_OPS4(float, Float)
TEST_COMPOSITE_OPS4(double, Double)
TEST_COMPOSITE_OPS4(int, Int)
//TEST_COMPOSITE_OPS4(ptMgrad::complex<float>, ComplexFloat)
//TEST_COMPOSITE_OPS4(ptMgrad::complex<double>, ComplexDouble)


#define TEST_COMPOSITE_OPS5(TYPE, NAME)                        \
	TEST(ValueTest, Composite##NAME##Operations5) {            \
		Value<TYPE> a = 2.0;                                   \
		Value<TYPE> b = 3.0;                                   \
		Value<TYPE> c = 4.0;                                   \
		Value<TYPE> d = 5.0;                                   \
                                                               \
		Value<TYPE> e = ptMgrad::pow(a + b, c) + 1;            \
                                                               \
		EXPECT_EQ(e.dataX(), TYPE(626.0));                     \
	}

TEST_COMPOSITE_OPS5(float, Float)
TEST_COMPOSITE_OPS5(double, Double)
TEST_COMPOSITE_OPS5(int, Int)
//TEST_COMPOSITE_OPS5(ptMgrad::complex<float>, ComplexFloat)
//TEST_COMPOSITE_OPS5(ptMgrad::complex<double>, ComplexDouble)


#define TEST_COMPOSITE_OPS6(TYPE, NAME)                        \
	TEST(ValueTest, Composite##NAME##Operations6) {            \
		Value<TYPE> a = 2.0;                                   \
		Value<TYPE> b = 3.0;                                   \
		Value<TYPE> c = 4.0;                                   \
		Value<TYPE> d = 5.0;                                   \
                                                               \
		Value<TYPE> e = d / c * 2.0 + 1;                       \
                                                               \
		EXPECT_EQ(e.dataX(), TYPE(3.5));                       \
	}

TEST_COMPOSITE_OPS6(float, Float)
TEST_COMPOSITE_OPS6(double, Double)
TEST_COMPOSITE_OPS6(int, Int)
//TEST_COMPOSITE_OPS6(ptMgrad::complex<float>, ComplexFloat)
//TEST_COMPOSITE_OPS6(ptMgrad::complex<double>, ComplexDouble)

