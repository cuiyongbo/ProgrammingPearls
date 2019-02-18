#include <iostream>
#include <vector>

using namespace std;

int main()
{
	vector<int> vi(10);
	cout << distance(vi.begin(), vi.end()) << '\n';
	return 0;
}

