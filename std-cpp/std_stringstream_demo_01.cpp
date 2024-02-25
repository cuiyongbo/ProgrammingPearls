#include <iostream>
#include <sstream>

using namespace std;

int main()
{
    string input1 = "hello,world,I,love,you";
    string input2 = "nice,to,meet,you";
    stringstream ss(input1);

    cout << "input position indicator: " <<  ss.tellg() << endl;
    cout << "output position indicator: " << ss.tellp() << endl;

    cout << ss.str() << endl;

    string item;
    while(std::getline(ss, item, ','))
    {
        cout << item << endl;
    }

    cout << "input position indicator: " <<  ss.tellg() << endl;
    cout << "output position indicator: " << ss.tellp() << endl;

    // reset stringstream's content, must clear previous state
    ss.clear();
    ss.str(input2);
    cout << ss.str() << endl;

    cout << "input position indicator: " <<  ss.tellg() << endl;
    cout << "output position indicator: " << ss.tellp() << endl;

    while(std::getline(ss, item, ','))
    {
        cout << item << endl;
    }
}
