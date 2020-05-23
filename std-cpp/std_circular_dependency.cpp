#include <iostream>
#include <memory>

#include "test_logger.h"

using namespace std;

void bad_case()
{
    TestLogger logger("bad_case");

    struct B;
    struct A
    {
        ~A() { cout << " ~A()" << endl;}
        shared_ptr<B> m_b;
    };
    struct B
    {
        ~B() { cout << " ~B()" << endl;}
        shared_ptr<A> m_a;
    };

    shared_ptr<A> a(new A);
    shared_ptr<B> b(new B);

    a->m_b = b;
    b->m_a = a;

    cout << "a.use_count(): " << a.use_count() << endl;
    cout << "b.use_count(): " << b.use_count() << endl;
}

void solution()
{
    TestLogger logger("circular dependency solution");

    struct B;
    struct A
    {
        ~A() { cout << " ~A()" << endl;}
        weak_ptr<B> m_b;
    };
    struct B
    {
        ~B() { cout << " ~B()" << endl;}
        weak_ptr<A> m_a;
    };

    shared_ptr<A> a(new A);
    shared_ptr<B> b(new B);

    a->m_b = b;
    b->m_a = a;

    cout << "a.use_count(): " << a.use_count() << endl;
    cout << "b.use_count(): " << b.use_count() << endl;
}

int main()
{
    bad_case();

    solution();
}
