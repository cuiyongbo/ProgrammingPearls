#include "test_logger.h"

using namespace std;

void test_polymorphism_badcase()
{
    TestLogger logger("test_polymorphism_badcase");
    class A
    {
    public:
        ~A() { cout << "~A" << endl;}
    };

    class B: public A
    {
    public:
        ~B() { cout << "~B" << endl;}
    };

    A* p = new B;
    delete p;
}

void test_polymorphism_basic()
{
    TestLogger logger("test_polymorphism_basic");

    class A
    {
    public:
        virtual ~A() { cout << "~A" << endl;}
    };

    class B: public A
    {
    public:
        ~B() { cout << "~B" << endl;}
    };

    A* p = new B;
    delete p;
}

int main()
{
    test_polymorphism_badcase();
    test_polymorphism_basic();
}
