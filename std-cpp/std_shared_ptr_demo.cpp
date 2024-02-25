#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>

using namespace std;

struct Base {
    Base() { std::cout << "  Base::Base()\n"; }
    virtual ~Base() { std::cout << "  Base::~Base()\n"; }
};

struct Derived: public Base {
    Derived() { std::cout << "  Derived::Derived()\n"; }
    ~Derived() { std::cout << "  Derived::~Derived()\n"; }
};

void thr(const std::shared_ptr<Base>& p) { // the signature make no difference
//void thr(std::shared_ptr<Base> p) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // thread-safe, even though the shared use_count is incremented
    std::shared_ptr<Base> lp = p;
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout << "local pointer in a thread:\n"
                  << "\tlp.get(): " << lp.get()
                  << ", lp.use_count(): " << lp.use_count() << "\n";
    }
}

void basic() {
    struct C {
        int* data;
    };
    shared_ptr<int> p1;
    shared_ptr<int> p2(nullptr);
    shared_ptr<int> p3(new int);
    shared_ptr<int> p4(new int, std::default_delete<int>());
    shared_ptr<int> p5(new int, [](int* p){ delete p; }, std::allocator<int>());
    shared_ptr<int> p6(p5);
    shared_ptr<int> p7(std::move(p6));
    p5 = p7;
    shared_ptr<int> p8(unique_ptr<int>(new int));
    shared_ptr<C> obj(new C);
    shared_ptr<int> p9(obj, obj->data);

    *p3 = 3;
    *p4 = 4;
    *p7 = 7; *p7 += 2;

    cout << "use_count:\n";
    cout << "\tp1: " << p1.use_count() << "\n";
    cout << "\tp2: " << p2.use_count() << "\n";
    cout << "\tp3: " << p3.use_count() << ", val: " << *p3 << "\n";
    cout << "\tp4: " << p4.use_count() << ", val: " << *p4 << "\n";
    cout << "\tp5: " << p5.use_count() << ", val: " << *p5 <<"\n";
    cout << "\tp6: " << p6.use_count() << "\n";
    cout << "\tp7: " << p7.use_count() << ", val: " << *p7 <<"\n";
    cout << "\tp8: " << p8.use_count() << "\n";
    cout << "\tp9: " << p9.use_count() << "\n";
}

void classicalUsage01() {
    using namespace std;

    struct Snack {
        int candy;
        Snack(int c):
            candy(c) {
        };
    };

    typedef shared_ptr<Snack> Snack_ptr;

    vector<Snack_ptr> sp;
    sp.push_back(Snack_ptr(new Snack(1)));
    sp.push_back(Snack_ptr(new Snack(2)));
    sp.push_back(Snack_ptr(new Snack(3)));
}

void classicalUsage02() {
class Foo {
public:
    Foo() {
        cout << "Foo()" << endl;
    }
    ~Foo() {
        cout << "~Foo()" << endl;
    }    
};

class D {
public:
    void operator() (Foo* p) {
        cout << "destructor with function object" << endl;
        delete p;
    }
};

    { // trivial case
        cout << "null shared_ptr" << endl;
        std::shared_ptr<Foo> o1;
    }

    {
        cout << "constuctor with object" << endl;
        std::shared_ptr<Foo> o1(new Foo);
        std::shared_ptr<Foo> o2(o1);
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;

        Foo* p = new Foo;
        o2.reset(p);
        cout << "after o2.reset" << endl;
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;
    }

    {
        std::shared_ptr<Foo> o1(new Foo, D());
        std::shared_ptr<Foo> o2(new Foo, [](Foo* p) {
            cout << "destructor with lambda function" << endl;
            delete p;
        });
    }
}

void multithread_case() {
    std::shared_ptr<Base> p = std::make_shared<Derived>();
    std::cout << "Create a shared Derived as a pointer to Base\n"
              << "\tp.get(): " << p.get()
              << ", p.use_count(): " << p.use_count() << '\n';

    std::thread t1(thr, p), t2(thr, p), t3(thr, p);
    p.reset();

    std::cout << "Shared ownership between 3 threads and released\n"
              << "ownership from main:\n"
              << "\tp.get(): " << p.get()
              << ", p.use_count(): " << p.use_count() << '\n';
    t1.join(); t2.join(); t3.join();
    std::cout << "All threads completed, the last one deleted\n";
}

void misc_case() {
    std::shared_ptr<int> p1(new int, std::default_delete<int>());
    std::shared_ptr<int> p2(new int[5], std::default_delete<int[]>());

    cout << "\tp1: " << p1.use_count() << "\n";
    cout << "\tp2: " << p2.use_count() << "\n";
}

int main() {
    basic();
	cout << "********************" << endl;
    classicalUsage01();
	cout << "********************" << endl;
    classicalUsage02();
	cout << "********************" << endl;
    multithread_case();
	cout << "********************" << endl;
    misc_case();
}
