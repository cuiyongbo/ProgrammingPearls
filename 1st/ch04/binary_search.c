#include <stdio.h>
#include "search_alg.h"

int binarySearch(int* arr, int start, int end, int val)
{
	if(start > end)
		return -1;

	while(start <= end)
	{
		int mid = start + (end - start)/2;
		if(arr[mid] == val)
			return mid;
		else if(arr[mid] > val)
			end = mid - 1;
		else
			start = mid + 1;
	}
	
	return -1;
}
