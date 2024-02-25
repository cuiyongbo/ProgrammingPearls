#include <iostream>

using namespace std;

void print() 
{
    cout << endl;
}

template <typename T> void print(const T& t) 
{
    cout << t << endl;
}

/*
    An ellipsis is used in two ways by variadic templates.
    To the left of the parameter name, it signifies a parameter pack, 
    and to the right of the parameter name, it expands the parameter packs into separate names.
*/

template <typename First, typename... Rest> void print(const First& first, const Rest&... rest)
{
    cout << first << ", ";
    print(rest...); // recursive call using pack expansion syntax
}

int main()
{
    print(); // calls first overload, outputting only a newline
    print(1); // calls second overload

    // these call the third overload, the variadic template,
    // which uses recursion as needed.
    print(10, 20);
    print(100, 200, 300);
    print("first", 2, "third", 3.14159);
}

/*
    Most implementations that incorporate variadic template functions use recursion of some form, 
    but it's slightly different from traditional recursion. Traditional recursion involves a function 
    calling itself by using the same signature. (It may be overloaded or templated, but the same signature 
    is chosen each time.) Variadic recursion involves calling a variadic function template by using differing
    (almost always decreasing) numbers of arguments, and thereby stamping out a different signature every time.
    A "base case" is still required, but the nature of the recursion is different.
*/
