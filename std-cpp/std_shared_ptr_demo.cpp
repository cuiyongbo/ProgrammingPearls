#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>

using namespace std;

int basic();
int classicalUsage();

struct Base
{
    Base() { std::cout << "  Base::Base()\n"; }
    // Note: non-virtual destructor is OK here
    ~Base() { std::cout << "  Base::~Base()\n"; }
};

struct Derived: public Base
{
    Derived() { std::cout << "  Derived::Derived()\n"; }
    ~Derived() { std::cout << "  Derived::~Derived()\n"; }
};

void thr(const std::shared_ptr<Base>& p)
{
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // thread-safe, even though the shared use_count is incremented
    std::shared_ptr<Base> lp = p;
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout << "local pointer in a thread:\n"
                  << "lp.get(): " << lp.get()
                  << ", lp.use_count(): " << lp.use_count() << "\n";
    }
}

int main()
{
    std::shared_ptr<Base> p = std::make_shared<Derived>();
    std::cout << "Create a shared Derived as a pointer to Base\n"
              << " p.get(): " << p.get()
              << ", p.use_count(): " << p.use_count() << '\n';

    std::thread t1(thr, p), t2(thr, p), t3(thr, p);
    p.reset();

    std::cout << "Shared ownership between 3 threads and released\n"
              << "ownership from main:\n"
              << " p.get(): " << p.get()
              << ", p.use_count(): " << p.use_count() << '\n';
    t1.join(); t2.join(); t3.join();
    std::cout << "All threads completed, the last one deleted\n";

    basic();
    classicalUsage();
}

int basic()
{
    struct C
    {
        int* data;
    };

    shared_ptr<int> p1;
    shared_ptr<int> p2(nullptr);
    shared_ptr<int> p3(new int);
    shared_ptr<int> p4(new int, std::default_delete<int>());
    shared_ptr<int> p5(new int, [](int* p){ delete p; }, std::allocator<int>());
    shared_ptr<int> p6(p5);
    shared_ptr<int> p7(std::move(p6));
    shared_ptr<int> p8(unique_ptr<int>(new int));
    shared_ptr<C> obj(new C);
    shared_ptr<int> p9(obj, obj->data);

    cout << "use_count:\n";
    cout << "p1: " << p1.use_count() << "\n";
    cout << "p2: " << p2.use_count() << "\n";
    cout << "p3: " << p3.use_count() << "\n";
    cout << "p4: " << p4.use_count() << "\n";
    cout << "p5: " << p5.use_count() << "\n";
    cout << "p6: " << p6.use_count() << "\n";
    cout << "p7: " << p7.use_count() << "\n";
    cout << "p8: " << p8.use_count() << "\n";
    cout << "p9: " << p9.use_count() << "\n";
    return 0;
}

int classicalUsage()
{
    using namespace std;

    struct Snack
    {
        int candy;

        Snack(int c):
            candy(c)
        {};
    };

    typedef shared_ptr<Snack> Snack_ptr;

    vector<Snack_ptr> sp;
    sp.push_back(Snack_ptr(new Snack(1)));
    sp.push_back(Snack_ptr(new Snack(2)));
    sp.push_back(Snack_ptr(new Snack(3)));
}
