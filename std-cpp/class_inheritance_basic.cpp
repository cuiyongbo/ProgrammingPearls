#include <iostream>

using namespace std;

class A  {
public:
    A() {cout << "A(): " << this << endl;}
    ~A() {cout << "~A(): " << this << endl;}
    A(const A& rhs) {cout << "A(const A&): " << this << endl; m_a = rhs.m_a;}
    A& operator=(const A& rhs) {cout << "operator=(const A&)\n"; m_a = rhs.m_a; return *this;}
    A(int a) { cout << "A(int): " << this << endl; m_a = a;}

private:
    int m_a;
};

class C {
public:
    C() {cout << "C():" << this << endl;}
    C(const C&) {cout << "C(const C&): " << this << endl;}
    ~C() {cout << "~C(): " << this << endl;}
};

class B : public A {
public:
    B() {cout << "B():" << this << endl;}
    B(const B&) {cout << "B(const B&): " << this << endl;}
    ~B() {cout << "~B(): " << this << endl;}
private:
    C m_c;
};

static C g_cc;

A test_return_arg_ref(A& aa) {
    A a = aa;
    return a;
}

A test_return_arg(A aa) {
    A a = aa;
    return a;
}

A test_return() {
    A a;
    return a;
}

void advanced_topics_01 () {
class Base {
public:
    static Base* create_instance() { return new Base;}
    ~Base() = default;
private:
    Base() = default;
    Base(const Base&) = delete;
    Base& operator=(const Base&) = delete;
};

/*
class Derived: public Base {
public:
    Derived() {}
};
    // Derived dd; // error: base class 'Base' has private default constructor
*/

    // Base bb; // "Base::Base()" (declared at line 53) is inaccessible
    // Base* bp = new Base; // "Base::Base()" (declared at line 53) is inaccessible

    Base* bp = Base::create_instance();
    delete bp;
}

void advanced_topics_02() {
/*
    class B {
        friend class C;
        int b;
    };
    class A {
        friend class B;
        int a;
        void f(B* p) {
            p->b++; // error: A is not a friend of B, despite B is a friend of A
        }
    };
    class C {
        void f(A* p) {
            p->a++; // error: C is not a friend of A, despite being a friend of a friend of A
        }
    };
    class D: public B {
        void f(A* p) {
            p->a++; // error: D is not a friend of A, despite being derived from a friend of A
        }
    };
*/
}


class TestBase {
public:
    void f() {}
    int a;
protected:
    int b;
private:
    int c;
};

class TestDerived: public TestBase {
public:
    void fd() {
        a++;
        b++;
        c++;
        printf("a=%d, b=%d\n", a, b);
    }
};


int main() {
    cout << "entering main" << endl;
    { TestDerived d; d.fd();}
    return 0;

    cout << "{ A a; A aa = test_return_arg_ref(a); }" << endl;
    { A a; A aa = test_return_arg_ref(a); }

    cout << "{ A a; A aa = test_return_arg(a); }" << endl;
    { A a; A aa = test_return_arg(a); }


    cout << "{ A a = test_return(); }" << endl;
    { A a = test_return(); }

    cout << "{ B bb; }" << endl;
    { B bb; }
    
    //exit(0);

    cout << "{ B b1; B b2=b1;}" << endl;
    { 
        B b1; 
        cout << "b1: " << &b1 << endl;
        B b2=b1;
        cout << "b2: " << &b2 << endl;
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
