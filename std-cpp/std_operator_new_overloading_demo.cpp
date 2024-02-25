#include <cstdio>
#include <cstdlib>
#include <iostream>

// Global replacement of a minimal set of functions:
void* operator new(std::size_t sz) {
    std::printf("global op new called, size = %zu\n",sz);
    return std::malloc(sz);
}
void operator delete(void* ptr) noexcept {
    std::puts("global op delete called");
    std::free(ptr);
}

// class-specific allocation functions
struct X {
    static void* operator new(std::size_t sz) {
        std::cout << "custom new for size " << sz << '\n';
        return ::operator new(sz);
    }
    static void* operator new[](std::size_t sz) {
        std::cout << "custom new for size " << sz << '\n';
        return ::operator new(sz);
    }
    int xx;
};

int main() {
    int* p1 = new int;
    delete p1;

    int* p2 = new int[10]; // guaranteed to call the replacement in C++11
    delete[] p2;

    X* x = new X;
    delete x;

    X* xs = new X[2];
    delete[] xs;
}


