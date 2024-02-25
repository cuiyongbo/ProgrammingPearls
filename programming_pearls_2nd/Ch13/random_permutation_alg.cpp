#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

int randInt(int l, int u) {return l+rand()%(u-l+1);}

void alg_1(int m, int n)
{
	std::vector<int> v;
	v.reserve(n);	
	for(int i=0; i<n+1; i++)
		v.push_back(i);

	for(int i=0; i<m; i++)
	{
		int j = randInt(i, n);
		std::swap(v[i], v[j]);		
	}

	for(int i=0; i<m; i++)
		std::cout << v[i] << " ";
	std::cout << "\n";
}

int main()
{
	srand((unsigned)time(NULL));
	
	int m, n;
	while(true)
	{
		std::cout << "Enter m element-permutation from [0, n): ";
		std::cin >> m >> n;
		alg_1(m , n);
	}

	return 0;
}

