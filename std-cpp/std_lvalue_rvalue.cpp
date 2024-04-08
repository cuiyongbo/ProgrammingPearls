#include <iostream>

// https://en.cppreference.com/w/cpp/language/value_category
// https://www.scs.stanford.edu/~dm/blog/decltype.html
// https://stackoverflow.com/questions/16637945/empirically-determine-value-category-of-c11-expression/16638081#16638081

decltype(auto)
fn_A(int i)
{
  return i;
}

decltype(auto)
fn_B(int i)
{
  return (i);
}

decltype(auto)
fn_C(int i)
{
  return (i+1);
}

decltype(auto)
fn_D(int i)
{
  return i++;
}

decltype(auto)
fn_E(int i)
{
  return ++i;
}

decltype(auto)
fn_F(int i)
{
  return (i >= 0 ? i : 0);
}

decltype(auto)
fn_G(int i, int j)
{
  return i >= j ? i : j;
}

struct S {
  int i = 0;
};

decltype(auto)
fn_H()
{
  return (S{});
}

decltype(auto)
fn_I()
{
  return (S{}.i);
}

template<typename T> constexpr const char *category = "prvalue";
template<typename T> constexpr const char *category<T&> = "lvalue";
template<typename T> constexpr const char *category<T&&> = "xvalue";

#define SHOW(E) std::cout << #E << ": " << category<decltype((E))> << std::endl

int main(int argc, char* argv[]) {
  SHOW(fn_A(0));
  SHOW(fn_B(0));
  SHOW(fn_C(0));
  SHOW(fn_D(0));
  SHOW(fn_E(0));
  SHOW(fn_F(0));
  SHOW(fn_G(0,1));
  SHOW(fn_H());
  SHOW(fn_I());
}

/*
# g++ std_lvalue_rvalue.cpp --std=c++14
std_lvalue_rvalue.cpp:14:11: warning: reference to stack memory associated with parameter 'i' returned [-Wreturn-stack-address]
  return (i);
          ^
std_lvalue_rvalue.cpp:44:19: warning: reference to stack memory associated with parameter 'i' returned [-Wreturn-stack-address]
  return i >= j ? i : j;
                  ^
std_lvalue_rvalue.cpp:44:23: warning: reference to stack memory associated with parameter 'j' returned [-Wreturn-stack-address]
  return i >= j ? i : j;
                      ^
std_lvalue_rvalue.cpp:60:11: warning: returning reference to local temporary object [-Wreturn-stack-address]
  return (S{}.i);

# ./a.out
fn_A(0): prvalue
fn_B(0): lvalue
fn_C(0): prvalue
fn_D(0): prvalue
fn_E(0): lvalue
fn_F(0): prvalue
fn_G(0,1): lvalue
fn_H(): prvalue
fn_I(): xvalue
*/


/* 
From chatGPT (回答不争取, 仅参考):

In C++, value categories are an important concept that determine how expressions can be used in code. There are three primary value categories:

1. **Lvalue (locator value)**: Refers to an expression that identifies a non-temporary object or a non-member function. Lvalues can appear on the left-hand side of an assignment expression.

2. **Prvalue (pure rvalue)**: Refers to an expression that identifies a temporary object or computes a value (not associated with any object storage). Prvalues can appear on the right-hand side of an assignment expression.

3. **Xvalue (eXpiring value)**: Refers to an expression that identifies an object that is about to be moved from, such as a temporary object or a cast to an rvalue reference type.

C++11 introduced two more general categories that encompass the above:

- **Glvalue (generalized lvalue)**: Includes both lvalues and xvalues.
- **Rvalue**: Includes both prvalues and xvalues.

To identify the value category of an expression or variable, you can use the following type traits and utility functions from the `<type_traits>` header:

- `std::is_lvalue_reference<T>::value`: Returns `true` if `T` is an lvalue reference type.
- `std::is_rvalue_reference<T>::value`: Returns `true` if `T` is an rvalue reference type.

Additionally, you can use the `decltype` specifier to determine the type of an expression and then apply these type traits. Here's an example:

```cpp
#include <iostream>
#include <type_traits>

int main() {
    int a = 5; // 'a' is an lvalue of type int
    int& l_ref = a; // 'l_ref' is an lvalue reference
    int&& r_ref = 5; // 'r_ref' is an rvalue reference

    // Check if 'a' is an lvalue
    std::cout << std::boolalpha << "Is 'a' an lvalue? " << std::is_lvalue_reference<decltype((a))>::value << std::endl;

    // Check if 'l_ref' is an lvalue
    std::cout << "Is 'l_ref' an lvalue? " << std::is_lvalue_reference<decltype((l_ref))>::value << std::endl;

    // Check if 'r_ref' is an rvalue
    std::cout << "Is 'r_ref' an rvalue? " << std::is_rvalue_reference<decltype((r_ref))>::value << std::endl;

    // Check if '5' is an rvalue
    std::cout << "Is '5' an rvalue? " << std::is_rvalue_reference<decltype((5))>::value << std::endl;

    // Check if "hello" is an lvalue/rvalue
    std::cout << "Is 'hello' an lvalue? " << std::is_lvalue_reference<decltype(("hello"))>::value << std::endl;
    std::cout << "Is 'hello' an rvalue? " << std::is_rvalue_reference<decltype(("hello"))>::value << std::endl;

    return 0;
}
```

In this example, `decltype((a))` is used to get the type of the expression `(a)`, which is an lvalue reference to `int`, hence `std::is_lvalue_reference<decltype((a))>::value` returns `true`. Note that the extra parentheses around `a` are necessary because `decltype(a)` would yield `int`, not `int&`.

For `r_ref`, which is an rvalue reference, `std::is_rvalue_reference<decltype((r_ref))>::value` returns `true`.

For the literal `5`, which is a prvalue, both `std::is_lvalue_reference<decltype((5))>::value` and `std::is_rvalue_reference<decltype((5))>::value` return `false` because it is not a reference at all; it's a prvalue.

Remember that these type traits check the type of the expression, not its value category directly. However, they can be used to infer the value category in many cases.
*/


/*
from chatGPT:

In C++, lvalues and rvalues are two of the main value categories, which classify expressions by their identities and lifetimes. Here's a brief explanation of each and how to distinguish them:

### Lvalue (Locator Value)

An **lvalue** refers to an expression that has an identifiable location in memory (i.e., it has an address). Lvalues can be used on the left-hand side of an assignment operator, meaning you can assign a value to them. Variables, references, and dereferenced pointers are typical examples of lvalues.

Characteristics of lvalues:
- They have an address that can be accessed.
- You can take their address using the address-of operator (`&`).
- They often represent objects that persist beyond a single expression.

Examples of lvalues:
- Named variables: `int x;`
- References: `int& ref = x;`
- String literals: `"Hello, World!"` (Note: String literals are lvalues, but they represent arrays of `const` characters, so you cannot assign to them.)
- Function calls that return lvalue references: `int& foo();`

### Rvalue (Read Value)

An **rvalue** refers to an expression that does not have an identifiable location in memory. Rvalues are typically temporary values that do not persist beyond the expression that uses them. They can be used on the right-hand side of an assignment operator. Literals (except string literals), arithmetic expressions, and function calls that return non-references are examples of rvalues.

Characteristics of rvalues:
- They do not have a stable address in memory.
- You cannot take their address (doing so is either illegal or doesn't make sense).
- They are temporary and often represent values that are not stored in an object.

Examples of rvalues:
- Literals (except string literals): `42`, `true`
- Arithmetic expressions: `x + y`
- Function calls that return non-references: `int foo();`

### Distinguishing Lvalues and Rvalues

Here's how you can distinguish between lvalues and rvalues:

- **Addressability**: If you can take the address of an expression using the `&` operator, it's likely an lvalue. If you cannot, it's likely an rvalue.
- **Assignment**: If you can assign to an expression, it's an lvalue. If you cannot, it's an rvalue.
- **Persistence**: If the expression refers to something that has a lasting identity beyond its immediate use, it's an lvalue. If it's a temporary result that doesn't have a lasting identity, it's an rvalue.

Here's a simple example to illustrate the difference:

```cpp
int x = 10; // 'x' is an lvalue
int y = x + 5; // 'x + 5' is an rvalue

int& l_ref = x; // OK: lvalue reference can bind to an lvalue
// int& l_ref_err = x + 5; // Error: lvalue reference cannot bind to an rvalue

int&& r_ref = x + 5; // OK: rvalue reference can bind to an rvalue
// int&& r_ref_err = x; // Error: rvalue reference cannot bind to an lvalue
```

In this example, `x` is an lvalue because it refers to a specific location in memory. The expression `x + 5` is an rvalue because it results in a temporary value that doesn't have a lasting identity. You can bind an lvalue reference to `x`, but not to `x + 5`. Conversely, you can bind an rvalue reference to `x + 5`, but not to `x`.
*/