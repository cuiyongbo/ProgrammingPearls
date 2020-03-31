#include <iostream>

using namespace std;

// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
class Singleton
{
public:
    static Singleton& getInstance()
    {
        // thread-safe in c++11
        static Singleton instance;
        return instance;
    }

    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

private:
    Singleton() {}
};

int main()
{
    Singleton& s1 = Singleton::getInstance();
    Singleton& s2 = Singleton::getInstance();
    cout << "s1: " << &s1 << "\n";
    cout << "s2: " << &s2 << "\n";
    return 0;
}
