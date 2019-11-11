#include <iostream>
#include <algorithm>
#include <string>
#include <cctype>
#include <vector>
#include <iterator>

using namespace std;

int main()
{
	string s("hello world");	
	transform(s.begin(), s.end(), s.begin(),
		[](unsigned char c) {return toupper(c);});

	vector<int> ordinals;
	transform(s.begin(), s.end(), back_inserter(ordinals),
		[](unsigned char c) { return c;});

	cout << s << ": ";
	copy(ordinals.begin(), ordinals.end(), ostream_iterator<int>(cout, " "));
	cout << "\n";
	return 0;
}
