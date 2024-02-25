#include <iostream>
#include <memory>

#include "test_logger.h"

using namespace std;

void case_study() {
    TestLogger logger("circular dependency case study");

    struct B;
    struct A {
        ~A() { cout << " ~A()" << endl;}
        shared_ptr<B> m_b;
    };
    struct B {
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

void solution() {
    TestLogger logger("circular dependency solution");

    struct B;
    struct A {
        ~A() { cout << " ~A()" << endl;}
        weak_ptr<B> m_b;
    };
    struct B {
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

void weak_ptr_classical_usage() {
    std::weak_ptr<int> wp_demo;
    auto observe = [&]() {
        cout << "use_count=" << wp_demo.use_count() << ": ";
        if (auto sp = wp_demo.lock()) {
            cout << *sp << endl;
        } else {
            cout << "wp_demo has expired" << endl;
        }
    };

    TestLogger logger(__func__);
    {
        auto sp = std::make_shared<int>(42);
        wp_demo = sp;
        observe();
    }
    observe();
}

int main() {
    {
        cout << "circular reference demo:" << endl;
        case_study();
        solution();
    }

    {
        weak_ptr_classical_usage();
    }
}
