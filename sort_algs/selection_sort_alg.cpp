#include "sort_algs.h"

void selectionSort(int* arr, int count)
{
	for(int i=0; i<count-1; i++)
	{
		int pos = i;
		for(int j=i+1; j<count; j++)
		{
			if(arr[pos] > arr[j])
				pos = j;
		}
		if(pos != i)
		{
			swapWithType(int, arr[pos], arr[i]);
		}
	}
}

void local_nth_element(int* a, int count, int k)
{
	for(int i=0; i<k; i++)
	{
		int pos = i;
		for(int j=i+1; j<count; j++)
		{
			if(arr[pos] > arr[j])
				pos = j;
		}
		if(pos != i)
		{
			swapWithType(int, arr[pos], arr[i]);
		}
	}	
	return a[k];
}
