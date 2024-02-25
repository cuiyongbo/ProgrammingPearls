#include <iostream>
#include <string>
#include <random>
#include <map>

using namespace std;

int main()
{
	random_device rn;
	uniform_int_distribution<int> dist(0, 9);
	
	cout << "Entropy: " << rn.entropy() << '\n';
	cout << "min: " << rn.min() << '\n';
	cout << "max: " << rn.max() << '\n';
	
	map<int, int> hist;
	for(int i=0; i<20000; ++i)
		++hist[dist(rn)];

/*
note: demo only: the performance of many 
implementations of random_device degrades sharply
once the entropy pool is exhausted. For practical use
random_device is generally only used to seed 
a PRNG such as mt19937
*/

	for(auto& p: hist)
	{
		cout << p.first << ": " << string(p.second/100, '*') << '\n';
	}
}
