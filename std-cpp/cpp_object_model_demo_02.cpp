#include <iostream>
using namespace std;

class EmptyClass {};

class ClassWithoutStaticObject
{
public:
    ClassWithoutStaticObject() = default;
    ~ClassWithoutStaticObject() = default;
private:
    int a;
};

class ClassWithStaticObject
{
public:
    static void foo()
    {
        cout << "ClassWithStaticObject::foo" << endl;
    }

private:
    int a;
    static int b;
};

int ClassWithStaticObject::b = 0;

class ClassWithStaticObjectDerived01: public ClassWithStaticObject {};

class ClassWithStaticObjectDerived02: public ClassWithStaticObject
{
public:
    static void foo()
    {
        cout << "ClassWithStaticObjectDerived02::foo" << endl;
        ClassWithStaticObject::foo();
    }
};

int main()
{
    cout << "sizeof(EmptyClass): " << sizeof(EmptyClass) << endl;
    cout << "sizeof(ClassWithStaticObject): " << sizeof(ClassWithStaticObject) << endl;
    cout << "sizeof(ClassWithoutStaticObject): " << sizeof(ClassWithoutStaticObject) << endl;

    ClassWithStaticObject::foo();
    ClassWithStaticObjectDerived01::foo();
    ClassWithStaticObjectDerived02::foo();
}