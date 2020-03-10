#include <iostream>
#include <set>
#include <string>

using namespace std;

int main()
{
	set<string> S;
	string t;
	while(cin >> t)
		S.insert(t);
	for (set<string>::iterator j=S.begin(); j != S.end(); ++j)
		cout << *j << '\n';
	return 0;
}

