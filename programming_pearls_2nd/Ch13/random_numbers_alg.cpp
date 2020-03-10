#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int randInt(int l, int u)
{
	return l+rand()%(u-l+1);
}

void alg_1(int m, int n)
{
	std::set<int> s;
	while(s.size()<m)
	{
		int j = randInt(1, n);
		if(s.find(j) == s.end())
			s.insert(j);
	}

	for(auto it=s.begin(); it != s.end(); ++it)
		std::cout << *it << " ";
	std::cout << "\n";
}

std::set<int> alg_2(int m, int n)
{
	std::set<int> s;
	s.clear();
	if(m == 0)
		return s;

	s = alg_2(m-1, n-1);
	int t = randInt(1, n);
	if(s.find(t) == s.end())
		s.insert(t);
	else 
		s.insert(n);
	return s;
}	

void alg_3(int m, int n)
{
	std::set<int>s;
	for(int i=n-m+1; i<=n; i++)
	{
		int t = randInt(1, i);
		if(s.find(t) == s.end())
			s.insert(t);
		else 
			s.insert(i);
	}

	for(auto it=s.begin(); it != s.end(); ++it)
		std::cout << *it << " ";
	std::cout << "\n";
}


int main()
{
	srand((unsigned)time(NULL));

	int m, n;
	while(true)
	{
		printf("Enter m intergers from [0, n]: ");
		scanf("%d %d", &m, &n);
		
		alg_3(m, n);
	}
	return 0;
}

