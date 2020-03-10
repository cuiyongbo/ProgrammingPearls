#include "sort_algs.h"
#include <stdlib.h>
#include <utility>

static int _getRandom(int low, int high)
{
	return low + rand()%(high-low+1);
}

// It will elicit the worst-case performance O(n^2) 
// in case all elements are equal. 
static int _partition(int* a, int lo, int hi)
{
	int pivot = a[hi]; // always choose the end as pivot
	int i=lo-1;
	for(int j=lo; j<hi; j++)
	{
		if(a[j] <= pivot)
		{
			i++;
			swapWithType(int, a[i], a[j]);
		}
	}
	swapWithType(int, a[i+1], a[hi]);
	return i+1;
}

static int _partition_AllElementsSame(int* a, int lo, int hi)
{
	int pivot = a[hi];
	int i = lo-1;
	int count = 0;
	for(int j=lo; j<hi; j++)
	{
		if(a[j] == pivot)
			count += 1;
		if(a[j] <= pivot)
		{
			i++;
			swapWithType(int, a[i], a[j]);
		}
	}
	swapWithType(int, a[i+1], a[hi]);
	return i+1 - count/2;
}

void quickSort(int* a, int lo, int hi)
{
	if(lo >= hi)
		return;

	int p = _partition_AllElementsSame(a, lo, hi);
	//int p = _partition(a, lo, hi);
	quickSort(a, lo, p-1);
	quickSort(a, p+1, hi);
}

static int _partition_hoare(int* a, int lo, int hi)
{
	int p = _getRandom(lo, hi);
	swapWithType(int, a[p], a[lo]);
	int x = a[lo];
	int i = lo - 1;
	int j = hi + 1;
	while(true)
	{
		while(true)
		{
			j--;
			if(a[j] <= x)
				break;
		}
		while(true)
		{
			i++;
			if(a[i] >= x)
				break;
		}
		if(i < j)
		{
			swapWithType(int, a[i], a[j]);
		}
		else 
			return j;
	}	
}

void quickSort_hoare(int* a, int lo, int hi)
{
	if (lo >= hi)
		return;

	int q = _partition_hoare(a, lo, hi);
	quickSort_hoare(a, lo, q);
	quickSort_hoare(a, q+1, hi);
}

static std::pair<int, int> _partition_threeWayPartition(int* a, int lo, int hi)
{
	int p = _getRandom(lo, hi);
	swapWithType(int, a[p], a[lo]);
	int x = a[lo];
	for(int i = lo; i <= hi ;)
	{
		if(a[i] < x)
		{
			swapWithType(int, a[i], a[lo]);
			i++; lo++;
		}
		else if(a[i] > x)
		{
			swapWithType(int, a[i], a[hi]);
			hi--;
		}
		else
			i++;
	}	
	return std::make_pair(lo, hi);
}

void quickSort_threeWayPartition(int* a, int lo, int hi)
{
	if (lo >= hi)
		return;

	std::pair<int, int> q = _partition_threeWayPartition(a, lo, hi);
	quickSort_threeWayPartition(a, lo, q.first - 1);
	quickSort_threeWayPartition(a, q.second+1, hi);
}

