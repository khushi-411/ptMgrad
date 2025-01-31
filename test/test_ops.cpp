#include <iostream>

#include <gtest/gtest.h>

#include "../src/engine.h"


using namespace ptMgrad;


// for float operations
// Value (op) Value = Value
TEST(ValueTest, BasicFloatOperations) {
    Value<float> a = 2.0f;
    Value<float> b = 3.0f;

    EXPECT_EQ(a.dataX(), 2.0f);
    EXPECT_EQ(b.dataX(), 3.0f);

    EXPECT_EQ((a + b).dataX(), 5.0f);
	(a + b).backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), 1.0f);

    EXPECT_EQ((a - b).dataX(), -1.0f);
	a.zero_grad();
	b.zero_grad();
	(a - b).backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), -1.0f);

    EXPECT_EQ((a * b).dataX(), 6.0f);
	a.zero_grad();
	b.zero_grad();
	(a * b).backward();
	EXPECT_EQ(a.gradX(), 3.0f);
	EXPECT_EQ(b.gradX(), 2.0f);

    EXPECT_EQ((a / b).dataX(), 2.0f / 3.0f);
	a.zero_grad();
	b.zero_grad();
	(a / b).backward();
	//EXPECT_NEAR(a.gradX(), 1.0f / 3.0f);   // TODO
	//EXPECT_EQ(b.gradX(), -2.0f / 9.0f);

    // using add, sub, mul, & div function
    EXPECT_EQ(add(a, b).dataX(), 5.0f);
	a.zero_grad();
	b.zero_grad();
	add(a, b).backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), 1.0f);

    EXPECT_EQ(sub(a, b).dataX(), -1.0f);
	a.zero_grad();
	b.zero_grad();
	sub(a, b).backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), -1.0f);

    EXPECT_EQ(mul(a, b).dataX(), 6.0f);
	a.zero_grad();
	b.zero_grad();
	mul(a, b).backward();
	EXPECT_EQ(a.gradX(), 3.0f);
	EXPECT_EQ(b.gradX(), 2.0f);

    EXPECT_NEAR(div(a, b).dataX(), 2.0f / 3.0f, 0.001);
	a.zero_grad();
	b.zero_grad();
	div(a, b).backward();
	//EXPECT_NEAR(a.gradX(), 1.0f / 3.0f);   // TODO
	//EXPECT_EQ(b.gradX(), -2.0f / 9.0f);

    // test addition
    Value<float> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0f);
	c.backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), 1.0f);

    // test substraction
    Value<float> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0f);
	a.zero_grad();
	b.zero_grad();
	d.backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), -1.0f);

    // test multiplication
    Value<float> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0f);
	a.zero_grad();
	b.zero_grad();
	e.backward();
	EXPECT_EQ(a.gradX(), 3.0f);
	EXPECT_EQ(b.gradX(), 2.0f);

    // test division
    Value<float> f = a / b;
    EXPECT_NEAR(f.dataX(), 2.0f / 3.0f, 0.001);
	a.zero_grad();
	b.zero_grad();
	f.backward();
	//EXPECT_NEAR(a.gradX(), 1.0f / 3.0f, 0.001);   // TODO
	//EXPECT_EQ(b.gradX(), -2.0f / 9.0f);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0f);
	a.zero_grad();
	b.zero_grad();
	a.backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), 0.0f);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0f);
	a.zero_grad();
	b.zero_grad();
	a.backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), 0.0f);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0f);
	a.zero_grad();
	b.zero_grad();
	a.backward();
	EXPECT_EQ(a.gradX(), 1.0f);
	EXPECT_EQ(b.gradX(), 0.0f);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 6.0f / 3.0f);
}


// for float operations
// Value (op) scalar = Value
TEST(ValueTest, BasicFloatOperationsWithScalar) {
    Value<float> a = 2.0f;
    float b = 3.0f;

    EXPECT_EQ(a.dataX(), 2.0f);
    EXPECT_EQ(b, 3.0f);

    EXPECT_EQ((a + b).dataX(), 5.0f);
    EXPECT_EQ((a - b).dataX(), -1.0f);
    EXPECT_EQ((a * b).dataX(), 6.0f);
    EXPECT_NEAR((a / b).dataX(), 2.0f / 3.0f, 0.001);

    // using add, sub, mul, & div function
    EXPECT_EQ(add(a, b).dataX(), 5.0f);
    EXPECT_EQ(sub(a, b).dataX(), -1.0f);
    EXPECT_EQ(mul(a, b).dataX(), 6.0f);
    EXPECT_NEAR(div(a, b).dataX(), 2.0f / 3.0f, 0.001);

    // test addition
    Value<float> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0f);

    // test substraction
    Value<float> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0f);

    // test multiplication
    Value<float> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0f);

    // test division
    Value<float> f = a / b;
    EXPECT_NEAR(f.dataX(), 2.0f / 3.0f, 0.001);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0f);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0f);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0f);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 6.0f / 3.0f);
}

/*
// for float operations
// scalar (op) Value = scalar
TEST(ValueTest, BasicScalarOperationsWithFloat) {
    float a = 2.0f;
    Value<float> b = 3.0f;

    EXPECT_EQ(a, 2.0f);
    EXPECT_EQ(b.dataX(), 3.0f);

    EXPECT_EQ(a + b.dataX(), 5.0f);
    EXPECT_EQ(a - b.dataX(), -1.0f);
    EXPECT_EQ(a * b.dataX(), 6.0f);
    EXPECT_NEAR(a / b.dataX(), 2.0f / 3.0f, 0.001);

    // using add, sub, mul, & div function
    EXPECT_EQ(add(a, b.dataX()), 5.0f);
    EXPECT_EQ(sub(a, b.dataX()), -1.0f);
    EXPECT_EQ(mul(a, b.dataX()), 6.0f);
    EXPECT_NEAR(div(a, b.dataX()), 2.0f / 3.0f, 0.001);

    // test addition
    float c = a + b.dataX();
    EXPECT_EQ(c, 5.0f);

    // test substraction
    float d = a - b.dataX();
    EXPECT_EQ(d, -1.0f);

    // test multiplication
    float e = a * b.dataX();
    EXPECT_EQ(e, 6.0f);

    // test division
    float f = a / b.dataX();
    EXPECT_NEAR(f, 2.0f / 3.0f, 0.001);

    // using assignment operator

    // test addition
    a += b.dataX();
    EXPECT_EQ(a, 5.0f);

    // test substraction
    a -= b.dataX();
    EXPECT_EQ(a, 2.0f);

    // test multiplication
    a *= b.dataX();
    EXPECT_EQ(a, 6.0f);

    // test division
    a /= b.dataX();
    EXPECT_EQ(a, 6.0f / 3.0f);
}
*/

// for float operations
// scalar (op) scalar = scalar
TEST(ValueTest, BasicScalarOperationsFloat) {
    float a = 2.0f;
    float b = 3.0f;

    EXPECT_EQ(a, 2.0f);
    EXPECT_EQ(b, 3.0f);

    EXPECT_EQ(a + b, 5.0f);
	// FIXME
	// (a + b).backward();
	// EXPECT_EQ(a.gradX(), 1.0f);
	// EXPECT_EQ(b.gradX(), 0.0f);

    EXPECT_EQ(a - b, -1.0f);
    EXPECT_EQ(a * b, 6.0f);
    EXPECT_NEAR(a / b, 2.0f / 3.0f, 0.001);

    // test addition
    float c = a + b;
    EXPECT_EQ(c, 5.0f);

    // test substraction
    float d = a - b;
    EXPECT_EQ(d, -1.0f);

    // test multiplication
    float e = a * b;
    EXPECT_EQ(e, 6.0f);

    // test division
    float f = a / b;
    EXPECT_NEAR(f, 2.0f / 3.0f, 0.001);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a, 5.0f);

    // test substraction
    a -= b;
    EXPECT_EQ(a, 2.0f);

    // test multiplication
    a *= b;
    EXPECT_EQ(a, 6.0f);

    // test division
    a /= b;
    EXPECT_EQ(a, 6.0f / 3.0f);
}


// for double operations
// Value (op) Value = Value
TEST(ValueTest, BasicDoubleOperations) {
    Value<double> a = 2.0;
    Value<double> b = 3.0;

    EXPECT_EQ(a.dataX(), 2.0);
    EXPECT_EQ(b.dataX(), 3.0);

    EXPECT_EQ((a + b).dataX(), 5.0);
    EXPECT_EQ((a - b).dataX(), -1.0);
    EXPECT_EQ((a * b).dataX(), 6.0);
    EXPECT_NEAR((a / b).dataX(), 2.0 / 3.0, 0.001);

    // test addition
    Value<double> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0);

    // test substraction
    Value<double> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0);

    // test multiplication
    Value<double> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0);

    // test division
    Value<double> f = a / b;
    EXPECT_NEAR(f.dataX(), 2.0 / 3.0, 0.001);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 6.0 / 3.0);
}


// for double operations
// Value (op) scalar = Value
TEST(ValueTest, BasicDoubleOperationsWithScalar) {
    Value<double> a = 2.0;
    double b = 3.0;

    EXPECT_EQ(a.dataX(), 2.0);
    EXPECT_EQ(b, 3.0);

    EXPECT_EQ((a + b).dataX(), 5.0);
    EXPECT_EQ((a - b).dataX(), -1.0);
    EXPECT_EQ((a * b).dataX(), 6.0);
    EXPECT_NEAR((a / b).dataX(), 2.0 / 3.0, 0.001);

    // test addition
    Value<double> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0);

    // test substraction
    Value<double> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0);

    // test multiplication
    Value<double> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0);

    // test division
    Value<double> f = a / b;
    EXPECT_NEAR(f.dataX(), 2.0 / 3.0, 0.001);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 6.0 / 3.0);
}


// for double operations
// scalar (op) Value = scalar
TEST(ValueTest, BasicScalarOperationsWithDouble) {
    double a = 2.0;
    Value<double> b = 3.0;

    EXPECT_EQ(a, 2.0);
    EXPECT_EQ(b.dataX(), 3.0);

    EXPECT_EQ(a + b.dataX(), 5.0);
    EXPECT_EQ(a - b.dataX(), -1.0);
    EXPECT_EQ(a * b.dataX(), 6.0);
    EXPECT_NEAR(a / b.dataX(), 2.0 / 3.0, 0.001);

    // test addition
    double c = a + b.dataX();
    EXPECT_EQ(c, 5.0);

    // test substraction
    double d = a - b.dataX();
    EXPECT_EQ(d, -1.0);

    // test multiplication
    double e = a * b.dataX();
    EXPECT_EQ(e, 6.0);

    // test division
    double f = a / b.dataX();
    EXPECT_NEAR(f, 2.0 / 3.0, 0.001);

    // using assignment operator

    // test addition
    a += b.dataX();
    EXPECT_EQ(a, 5.0);

    // test substraction
    a -= b.dataX();
    EXPECT_EQ(a, 2.0);

    // test multiplication
    a *= b.dataX();
    EXPECT_EQ(a, 6.0);

    // test division
    a /= b.dataX();
    EXPECT_EQ(a, 6.0 / 3.0);
}


// for double operations
// scalar (op) scalar = scalar
TEST(ValueTest, BasicScalarOperationsWithScalar) {
    double a = 2.0;
    double b = 3.0;

    EXPECT_EQ(a, 2.0);
    EXPECT_EQ(b, 3.0);

    EXPECT_EQ(a + b, 5.0);
    EXPECT_EQ(a - b, -1.0);
    EXPECT_EQ(a * b, 6.0);
    EXPECT_NEAR(a / b, 2.0 / 3.0, 0.001);

    // test addition
    double c = a + b;
    EXPECT_EQ(c, 5.0);

    // test substraction
    double d = a - b;
    EXPECT_EQ(d, -1.0);

    // test multiplication
    double e = a * b;
    EXPECT_EQ(e, 6.0);

    // test division
    double f = a / b;
    EXPECT_NEAR(f, 2.0 / 3.0, 0.001);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a, 5.0);

    // test substraction
    a -= b;
    EXPECT_EQ(a, 2.0);

    // test multiplication
    a *= b;
    EXPECT_EQ(a, 6.0);

    // test division
    a /= b;
    EXPECT_EQ(a, 6.0 / 3.0);
}


// for float x double operations
// FloatValue (op) DoubleValue = DoubleValue
TEST(ValueTest, BasicFloatDoubleOperations) {
    Value<float> a = 2.0f;
    Value<double> b = 3.0;

    EXPECT_EQ(a.dataX(), 2.0f);
    EXPECT_EQ(b.dataX(), 3.0);

    EXPECT_EQ((a + b).dataX(), 5.0);
	//(a + b).backward();
	//EXPECT_EQ(a.gradX(), 1.0f);
	//EXPECT_EQ(b.gradX(), 1.0f);

    EXPECT_EQ((a - b).dataX(), -1.0);
    EXPECT_EQ((a * b).dataX(), 6.0);
    EXPECT_NEAR((a / b).dataX(), 2.0 / 3.0, 0.001);

    // test addition
    Value<double> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0);

    // test substraction
    Value<double> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0);

    // test multiplication
    Value<double> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0);

    // test division
    Value<double> f = a / b;
    EXPECT_NEAR(f.dataX(), 2.0 / 3.0, 0.001);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 6.0 / 3.0);
}

// for double x float operations
// DoubleValue (op) FloatValue = DoubleValue
TEST(ValueTest, BasicDoubleFloatOperations) {
    Value<double> a = 2.0;
    Value<float> b = 3.0f;

    EXPECT_EQ(a.dataX(), 2.0);
    EXPECT_EQ(b.dataX(), 3.0f);

    EXPECT_EQ((a + b).dataX(), 5.0);
    EXPECT_EQ((a - b).dataX(), -1.0);
    EXPECT_EQ((a * b).dataX(), 6.0);
    EXPECT_NEAR((a / b).dataX(), 2.0 / 3.0, 0.001);

    // test addition
    Value<double> c = a + b;
    EXPECT_EQ(c.dataX(), 5.0);

    // test substraction
    Value<double> d = a - b;
    EXPECT_EQ(d.dataX(), -1.0);

    // test multiplication
    Value<double> e = a * b;
    EXPECT_EQ(e.dataX(), 6.0);

    // test division
    Value<double> f = a / b;
    EXPECT_NEAR(f.dataX(), 2.0 / 3.0, 0.001);

    // using assignment operator

    // test addition
    a += b;
    EXPECT_EQ(a.dataX(), 5.0);

    // test substraction
    a -= b;
    EXPECT_EQ(a.dataX(), 2.0);

    // test multiplication
    a *= b;
    EXPECT_EQ(a.dataX(), 6.0);

    // test division
    a /= b;
    EXPECT_EQ(a.dataX(), 6.0 / 3.0);
}


// int main(int argc, char **argv) {
//     testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }