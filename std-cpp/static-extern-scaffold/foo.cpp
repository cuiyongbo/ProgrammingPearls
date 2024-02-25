#include "glue.h"

using namespace std;

void foo()
{
    cout << "foo: g_static_sugar address: " << &g_static_sugar << ", value: " << g_static_sugar << endl;
    cout << "foo: g_extern_sugar address: " << &g_extern_sugar << ", value: " << g_extern_sugar << endl;
}