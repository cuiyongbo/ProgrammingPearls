#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <algorithm>

int bigrand()
{
	return RAND_MAX * rand() + rand();
}

int randint(int l, int u)
{
	return l + bigrand() % (u-l+1);
}

void genknuth(int m, int n)
{
	for(int i=0; i<n; i++)
	{
		if(bigrand() % (n-i) < m)
		{
			printf("%d\n", i);
			m--;
		}
	}
}

void gensets(int m, int n)
{
	std::set<int> s;
	while(s.size() < m)
		s.insert(bigrand() % n);
	std::set<int>::iterator i;
	for(i=s.begin(); i != s.end(); ++i)
		printf("%d\n", *i);
}

void genshuf(int m, int n)
{
	int* x = new int[n];
	for(int i=0; i<n; i++)
		x[i] = i;
	for(int i=0; i<m; i++)
	{
		int j = randint(i, n-1);
		int t = x[i]; x[i] = x[j]; x[j] = t;
	}
	std::sort(x, x+m);

	for(int i=0; i<m; i++)
		printf("%d\n", x[i]);
}

