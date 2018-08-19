#include "sort_algs.h"

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

