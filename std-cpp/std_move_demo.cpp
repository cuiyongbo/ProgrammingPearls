#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    vector<string> v;
    string s = "hello world";

    v.push_back(s);
    cout << "After copy, str is \"" << s << "\"\n";

    v.push_back(move(s));
    cout << "After move, str is \"" << s << "\"\n";

    cout << "The content of the vector:\n";
    cout << v[0] << endl;
    cout << v[1] << endl;

    return 0;
}

