#include "glue.h"

using namespace std;

int g_extern_sugar;

int main()
{
    g_extern_sugar = 3;
    g_static_sugar = 3;

    cout << "main: g_static_sugar address: " << &g_static_sugar << ", value: " << g_static_sugar << endl;
    cout << "main: g_extern_sugar address: " << &g_extern_sugar << ", value: " << g_extern_sugar << endl;

    foo();
    bar();
}