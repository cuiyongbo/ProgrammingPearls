// https://en.cppreference.com/w/cpp/language/move_constructor
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

struct A {
    std::string s;
    int k;
 
    A() : s("test"), k(-1) {}
    A(const A& o) : s(o.s), k(o.k) { std::cout << "move failed!\n"; }
    A(A&& o) noexcept :
        s(std::move(o.s)),       // explicit move of a member of class type
        k(std::exchange(o.k, 0)) // explicit move of a member of non-class type
    {}
};
 
A f(A a) {
    return a;
}
 
struct B : A {
    std::string s2;
    int n;
    // implicit move constructor B::(B&&)
    // calls A's move constructor
    // calls s2's move constructor
    // and makes a bitwise copy of n
};
 
struct C : B {
    ~C() {} // destructor prevents implicit move constructor C::(C&&)
};
 
struct D : B {
    D() {}
    ~D() {}           // destructor would prevent implicit move constructor D::(D&&)
    D(D&&) = default; // forces a move constructor anyway
};
 
int main() {
    std::cout << "Trying to move A\n";
    A a1 = f(A()); // return by value move-constructs the target
                   // from the function parameter
 
    std::cout << "Before move, a1.s = " << std::quoted(a1.s)
        << " a1.k = " << a1.k << '\n';
 
    A a2 = std::move(a1); // move-constructs from xvalue
    std::cout << "After move, a1.s = " << std::quoted(a1.s)
        << " a1.k = " << a1.k << '\n';
 
    std::cout << "\nTrying to move B\n";
    B b1;
 
    std::cout << "Before move, b1.s = " << std::quoted(b1.s) << "\n";
 
    B b2 = std::move(b1); // calls implicit move constructor
    std::cout << "After move, b1.s = " << std::quoted(b1.s) << "\n";
 
    std::cout << "\nTrying to move C\n";
    C c1;
    C c2 = std::move(c1); // calls copy constructor
 
    std::cout << "\nTrying to move D\n";
    D d1;
    D d2 = std::move(d1);
}

/*
# g++ std_move_constructor.cpp -std=c++14
# ./a.out
Trying to move A
Before move, a1.s = "test" a1.k = -1
After move, a1.s = "" a1.k = 0

Trying to move B
Before move, b1.s = "test"
After move, b1.s = ""

Trying to move C
move failed!

Trying to move D
*/

/*
A move constructor in C++ is a special member function that moves an object rather than copying it. This is particularly useful for objects that manage resources like dynamic memory, file handles, or network connections, where copying the resource can be expensive or undesirable.

### How Move Constructors Work

A move constructor takes an rvalue reference to an object of the same class as its parameter. This allows the constructor to transfer the resources from the source object to the newly created object, leaving the source in a valid but unspecified state. The syntax for a move constructor looks like this:

```cpp
class ClassName {
public:
    ClassName(ClassName&& other);
};
```

Here, `ClassName&&` is an rvalue reference to `ClassName`, which matches temporary objects and objects explicitly cast to `ClassName&&`.

### When to Use a Move Constructor

You should use a move constructor when you want to avoid the overhead of copying an object's resources, and you know that the source object will not be used again in its current state. This is often the case when:

- Returning large objects from functions by value.
- Storing objects in containers that may reallocate, such as `std::vector`.
- Swapping the contents of two objects.

### How to Implement a Move Constructor

To implement a move constructor, you typically perform the following steps:

1. Acquire the resources from the source object. This can often be done by simply copying the pointers or handles to the resources.
2. Set the source object's pointers or handles to null or some other safe state, so that the destructor of the source object does not free the resources now owned by the new object.

Here's an example of a simple move constructor for a class that manages a dynamic array:

```cpp
class DynamicArray {
public:
    // Regular constructor
    DynamicArray(size_t size) : size_(size), data_(new int[size]) {}

    // Move constructor
    DynamicArray(DynamicArray&& other) noexcept
        : size_(other.size_), data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr; // Prevent the destructor from deleting the data
    }

    // Destructor
    ~DynamicArray() {
        delete[] data_;
    }

    // Other members...

private:
    size_t size_;
    int* data_;
};
```

In this example, the move constructor takes the `size_` and `data_` from the `other` object, and then sets `other.size_` to `0` and `other.data_` to `nullptr`. This ensures that when the `other` object is destroyed, its destructor won't delete the memory that has been moved to the new object.

### How to Use a Move Constructor

You can invoke a move constructor explicitly by using `std::move` to cast an lvalue to an rvalue reference, signaling that you allow the object to be moved:

```cpp
DynamicArray a(100); // Regular constructor
DynamicArray b(std::move(a)); // Move constructor is called
```

After this operation, `a` is in a valid but unspecified state. You should not use `a` except to assign a new value to it or to destroy it.

The move constructor is also invoked implicitly in certain situations, such as when returning a local object by value from a function or when resizing a `std::vector` that stores objects of the class.

By using move semantics, you can write more efficient C++ code that avoids unnecessary copies of objects, especially when dealing with resources or large data structures.
*/