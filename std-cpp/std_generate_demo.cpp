#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>

using namespace std;

int f()
{
	static int i = 1;
	return i++;
}

void printVector(const char* msg, vector<int>& v)
{
	cout << msg << ":\n";
	copy(v.begin(), v.end(), ostream_iterator<int>(cout, " "));
	cout << "\n";
}

int main()
{
	vector<int> v(5, 0);
	generate(v.begin(), v.end(), f);
	printVector("generate(f)", v);

	generate(v.begin(), v.end(), [n=0] () mutable {return n++;});
	printVector("generate(lambda)", v);
}
