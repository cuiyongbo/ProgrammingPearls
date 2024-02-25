#include <iostream>
#include <map>
#include <string>

using namespace std;

int main()
{
	map<string, int> M;
	string t;
	while(cin >> t)
		M[t]++;
	for (map<string, int>::iterator j=M.begin(); j != M.end(); ++j)
		cout << j->first << " " << j->second << '\n';
	return 0;
}

