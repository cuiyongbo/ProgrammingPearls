#include "leetcode.h"

using namespace std;

void printVector(vector<int>& input)
{
	copy(input.begin(), input.end(), ostream_iterator<int>(cout, " "));
	cout << "\n";
}
