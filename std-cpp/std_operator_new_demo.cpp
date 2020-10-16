#include <iostream>
#include <new>

// https://isocpp.org/wiki/faq/freestore-mgmt#num-elems-in-new-array

using namespace std;

class ZeroedObject
{
public:
    ZeroedObject() { cout << "ZeroedObject()\n"; }
    ~ZeroedObject() { cout << "~ZeroedObject()\n"; }

public:
    void* operator new(size_t size);
    void operator delete(void* p);
    void* operator new[](size_t size);
    void operator delete[](void* p);
};

void* ZeroedObject::operator new(size_t size)
{
    cout << "operator new\n";
    void* ptr = NULL;
    if (size != 0)
    {
        ptr = ::operator new(size);
        if(ptr == NULL)
            throw std::bad_alloc();
        // throw bad_alloc if failed
    }
    return ptr;
}

void ZeroedObject::operator delete(void* p)
{
    cout << "operator delete\n";
    ::operator delete(p);
}

void* ZeroedObject::operator new[](size_t size)
{
    cout << "operator new[] " << size << "\n";
    void* ptr = NULL;
    if (size != 0)
    {
        ptr = ::operator new[](size);
        if(ptr == NULL)
            throw std::bad_alloc();
        // throw bad_alloc if failed
    }
    return ptr;
}

void ZeroedObject::operator delete[](void* p)
{
    cout << "operator delete[]\n";
    ::operator delete[](p);
}

void test_1()
{
    ZeroedObject* pp1 = new ZeroedObject[4];
    delete[] pp1;

    ZeroedObject* pp = new ZeroedObject;
    delete pp;
}

struct X {
    static void* operator new(std::size_t sz)
    {
        std::cout << "custom new for size " << sz << '\n';
        return ::operator new(sz);
    }
    static void* operator new[](std::size_t sz)
    {
        std::cout << "custom new[] for size " << sz << '\n';
        return ::operator new(sz);
    }
    static void operator delete(void* ptr, std::size_t sz)
    {
        std::cout << "custom delete for size " << sz << '\n';
        ::operator delete(ptr);
    }
    static void operator delete[](void* ptr, std::size_t sz)
    {
        std::cout << "custom delete for size " << sz << '\n';
        ::operator delete(ptr);
    }
};

void test_2() {
    X* p1 = new X;
    delete p1;
    X* p2 = new X[10];
    delete[] p2;

    p2 = new X[13];
    delete[] p2;
}

int main() {
    test_1();
    test_2();
}
