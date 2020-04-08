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

int main()
{
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
}
