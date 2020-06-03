#include "hello.pb.h"

// compile with:  g++ main.cpp hello.pb.cc -lprotobuf


using namespace std;

int main()
{
    hello demo;
    cout << demo.data() << endl;

    demo.set_data("How do you do?");
    cout << demo.data() << endl;
}