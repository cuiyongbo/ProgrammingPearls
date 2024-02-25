#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif

using namespace std;

namespace ans1
{
    template <class T>
    std::string type_name()
    {
        typedef typename std::remove_reference<T>::type TR;
        std::unique_ptr<char, void(*)(void*)> own
               (
    #ifndef _MSC_VER
                    abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                               nullptr, nullptr),
    #else
                    nullptr,
    #endif
                    std::free
               );
        std::string r = own != nullptr ? own.get() : typeid(TR).name();
        if (std::is_const<TR>::value)
            r += " const";
        if (std::is_volatile<TR>::value)
            r += " volatile";
        if (std::is_lvalue_reference<T>::value)
            r += "&";
        else if (std::is_rvalue_reference<T>::value)
            r += "&&";
        return r;
    }
}

namespace ans2
{
    template<class T> 
    std::string type_name(T var)
    {
        static const size_t char_hash_code = typeid(char).hash_code();
        static const size_t int_hash_code = typeid(int).hash_code();
        static const size_t float_hash_code = typeid(float).hash_code();
        static const size_t double_hash_code = typeid(double).hash_code();
        static const size_t string_hash_code = typeid(std::string).hash_code();
        size_t var_hash_code = typeid(T).hash_code();

        typedef typename std::remove_reference<T>::type TR;

        if(var_hash_code == char_hash_code)
            r = "char";
        else if(var_hash_code == int_hash_code)
            r = "int";
        else if(var_hash_code == float_hash_code)
            r = "float";
        else if(var_hash_code == double_hash_code)
            r = "double";
        else if(var_hash_code == string_hash_code)
            r = "string";
        else
            return "unknown type";

        if (std::is_const<TR>::value)
            r += " const";
        if (std::is_volatile<TR>::value)
            r += " volatile";
        if (std::is_lvalue_reference<T>::value)
            r += "&";
        else if (std::is_rvalue_reference<T>::value)
            r += "&&";
        
        return r;
    }
}

int main()
{
    int c = 1;
    cout << "int c : " << typeid(c).name() << endl;

    const int a = 1;
    cout << "const int a : " << typeid(a).name() << endl;

    cout << "type_name<decltype(c)>: " << ans1::type_name<decltype(c)>() << endl;
    cout << "type_name<decltype(a)>: " << ans1::type_name<decltype(a)>() << endl;

    cout << "type_name<decltype(c)>: " << ans2::type_name(c) << endl;
    cout << "type_name<decltype(a)>: " << ans2::type_name(a) << endl;

}