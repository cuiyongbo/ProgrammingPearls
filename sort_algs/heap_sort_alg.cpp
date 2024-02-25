#include "sort_algs.h"

static void _siftDownByLoop(int* a, int p, int count)
{
	while(p < count)
	{
		int largest = p;
		int left = 2*largest + 1;
		int right = left + 1;	
		if(left < count && a[left] > a[largest])
			largest  = left;
		if(right < count && a[right] > a[largest])
			largest = right;

		if(largest == p)
			break;

		swapWithType(int, a[p], a[largest]);
		p = largest;
	}
}

/*
	pre: arr[p+1, count) is alread a heap
	post: arr[p, count] is a heap
*/
static void _siftDown(int* a, int p, int count)
{
	int largest = p;
	int left = 2*p + 1;
	int right = left + 1;
	if(left < count && a[left] > a[largest])
		largest = left;
	if(right < count && a[right] > a[largest])
		largest = right;
	if(largest != p)
	{
		swapWithType(int, a[largest], a[p]);
		_siftDown(a, largest, count);
	}
}

void heapSort_siftDown(int* a, int count)
{
	// Build max heap
	for(int i=count/2; i>=0; i--)
		_siftDown(a, i, count);
	
	// Loop until one element left,
	// where it is naturally sorted.
	for(int i=count-1; i>0; i--)
	{
		// sorted array +1
		swapWithType(int, a[i], a[0]);

	 	// heap size -1
		// re-establish heap property
		_siftDown(a, 0, i);
	}
}

/*
	pre: a[0, i-1] is alread a heap
	post: a[0, i] is a heap
*/
static void _siftUp(int* a, int start, int end)
{
	for(int i=end; i != start;)
	{
		int p = (i - 1)/2; // parent index of node i
		if(a[p] < a[i])
		{
			swapWithType(int, a[p], a[i]);
			i = p;
		}
		else
			break; // array [0, i-1] is already a heap
	}	
}

void heapSort_siftUp(int* a, int count)
{
	for(int i=1; i<count; i++)
		_siftUp(a, 0, i);

	for(int i = count-1; i>0; i--)
	{
		swapWithType(int, a[i], a[0]);
		_siftDownByLoop(a, 0, i);
	}
}
