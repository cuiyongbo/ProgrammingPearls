#include "search_alg.h"

int binarySearch(int* arr, int size, int val)
{
	int start = 0;
	int end = size - 1;
	if (start > end || arr[start] > val || arr[end] < val)
		return -1;

	while (start <= end)
	{
		int mid = start + (end - start) / 2;
		if (arr[mid] == val)
			return mid;
		else if (arr[mid] > val)
			end = mid - 1;
		else
			start = mid + 1;
	}
	return -1;
}

int lowerBound(int* arr, int size, int val)
{
	int l = 0;
	int r = size;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (arr[mid] < val)
			l = mid + 1;
		else
			r = mid;
	}
	return l;
}

int upperBound(int* arr, int size, int val)
{
	int l = 0;
	int r = size;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (arr[mid] <= val)
			l = mid + 1;
		else
			r = mid;
	}
	return l-1;
}

