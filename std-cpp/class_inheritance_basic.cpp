#include <iostream>

using namespace std;

class A 
{
public:
    A() {cout << "A()\n";}
    ~A() {cout << "~A()\n";}
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
    ~C() {cout << "~C()\n";}
};

class B : public A
{
public:
    B() {cout << "B()\n";}
    ~B() {cout << "~B()\n";}
private:
    C m_c;
};

int main()
{
    { B bb; }
    cout << "=============\n";
    { A a1(1); }
    cout << "=============\n";
    { A a2 = 2;}
    cout << "=============\n";
    {A a1; A a2(a1);}
    cout << "=============\n";
    {A a1; A a2 = a1;}
    cout << "=============\n";
    {A a1; A a2; a2 = a1;}
}
