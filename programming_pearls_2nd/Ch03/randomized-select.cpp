#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <stdint.h>

using namespace std;

class Solution
{
	int partition(int arr[], int l, int u)
	{
		int m = l;
		int x = arr[u]; 
		for(int j=l; j<=u-1; j++)
		{
			if(arr[j] <= x)
			{
				swap(arr[m], arr[j]);
				m++;
			}
		}
		swap(arr[m], arr[u]);
		return m;
	}

	int randomPartition(int arr[], int l, int u)
	{
		int n = u-l+1;
		int pivot = rand()%n;
		swap(arr[l+pivot], arr[u]);
		return partition(arr, l, u);
	}

public:
	
	int kthSmallest(int arr[], int l, int u, int k)
	{
		while(k>0 && k<=u-l+1)
		{
			int q = randomPartition(arr, l, u);
			int m = q-l+1;
			if(k == m)
			{
				return arr[q];
			}
			else if(k < m)
			{
				u = q-1;
			}
			else
			{
				k = k-m;
				l = q+1;
			}
		}
		return INT32_MAX;
	}
};

int main()
{
	int arr[7] = {10, 100, 2, 4, 1, -2, 8};
	Solution s;
    assert(s.kthSmallest(arr, 0, 6, 1) == -2);
    assert(s.kthSmallest(arr, 0, 6, 2) == 1);
    assert(s.kthSmallest(arr, 0, 6, 3) == 2);
    assert(s.kthSmallest(arr, 0, 6, 4) == 4);
    assert(s.kthSmallest(arr, 0, 6, 5) == 8);
	assert(s.kthSmallest(arr, 0, 6, 6) == 10);
    assert(s.kthSmallest(arr, 0, 6, 7) == 100);

	return 0;
}
