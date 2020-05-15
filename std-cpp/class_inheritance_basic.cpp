#include <iostream>

using namespace std;

class A 
{
public:
    A() {cout << "A()\n";}
    ~A() {cout << "~A(this = " << (size_t)this << ")\n";}
    A(const A& rhs) {cout << "A(const A&)\n"; m_a = rhs.m_a;}
    A& operator=(const A& rhs) {cout << "operator=(const A&)\n"; m_a = rhs.m_a; return *this;}
    A(int a) { cout << "A(int)\n"; m_a = a;}

private:
    int m_a;
};

class C
{
public:
    C() {cout << "C()\n";}
    ~C() {cout << "~C(this = " << (size_t)this << ")\n";}
};

class B : public A
{
public:
    B() {cout << "B()\n";}
    ~B() {cout << "~B(this = " << (size_t)this << ")\n";}
private:
    C m_c;
};

static C g_cc;

A test_return_arg_ref(A& aa)
{
    A a = aa;
    return a;
}

A test_return_arg(A aa)
{
    A a = aa;
    return a;
}

A test_return()
{
    A a;
    return a;
}

int main()
{
    cout << "entering main" << endl;

    cout << "{ A a; A aa = test_return_arg_ref(a); }" << endl;
    { A a; A aa = test_return_arg_ref(a); }

    cout << "{ A a; A aa = test_return_arg(a); }" << endl;
    { A a; A aa = test_return_arg(a); }

    exit(0);

    cout << "{ A a = test_return(); }" << endl;
    { A a = test_return(); }

    cout << "{ B bb; }" << endl;
    { B bb; }

    cout << "{ B b1; B b2=b1;}" << endl;
    { 
        B b1; 
        cout << "b1: " << (size_t)&b1 << endl;
        B b2=b1;
        cout << "b2: " << (size_t)&b2 << endl;
    }

    cout << "{ A a1(1); }" << endl;
    { A a1(1); }
    
    cout << "{ A a2 = 2;}" << endl;
    { A a2 = 2;}

    cout << "{A a1; A a2(a1);}" << endl;
    {A a1; A a2(a1);}

    cout << "{A a1; A a2 = a1;}\n";
    {A a1; A a2 = a1;}
    
    cout << "{A a1; A a2; a2 = a1;}" << endl;
    {A a1; A a2; a2 = a1;}

    cout << "exiting main" << endl;
}
