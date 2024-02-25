#include <stdio.h>
#include "sort_algs.h"

void insertionSort_noSwap(int* a, int count)
{
	int i=1;
	for(; i<count; i++)
	{
		int val = a[i];
		int j = i;
		for(; j>0 && a[j-1]>val; j--)
			a[j] = a[j-1];
		a[j] = val;
	}
}

void insertionSort(int* a, int count)
{
	int i = 1;
	for(; i<count; i++)
	{
		int j=i;
		for(; j>0 && a[j-1]>a[j]; j--)
			swapWithType(int, a[j], a[j-1]);
	}
}

