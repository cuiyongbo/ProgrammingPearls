#include <iostream>
#include <set>

using namespace std;

int main()
{       
	set<int> S;

	int i;
	while (cin >> i)
	        S.insert(i);

	set<int>::iterator j;
	for (j = S.begin(); j != S.end(); ++j)
	        cout << *j << "\n";

	return 0;
}
