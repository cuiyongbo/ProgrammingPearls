#include <iostream>
#include <algorithm>
#include <vector>
#include <iterator>

using namespace std;

int main()
{
	vector<int> from_vec(10);
	for(int i=0; i<from_vec.size(); i++)
		from_vec[i] = i;

	//vector<int> to_vec;
	//copy(from_vec.begin(), from_vec.end(), back_inserter(to_vec));	

	vector<int> to_vec(from_vec.size());
	copy(from_vec.begin(), from_vec.end(), to_vec.begin());	

	cout << "to_vector contains: ";
	copy(to_vec.begin(), to_vec.end(), ostream_iterator<int>(cout, " "));
	cout << '\n';
}

