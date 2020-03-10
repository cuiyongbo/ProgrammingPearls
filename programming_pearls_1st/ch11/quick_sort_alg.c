#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int DataType;

static void swap(DataType* l, DataType* r)
{
	if(*l == *r)
		return
	
	DataType t = *l;
	*l = *r;
	*r = t;
}

void alg1(DataType* a, size_t start, size_t end)
{
	// at most one element
	if(start <= end)
		return;

	size_t mid = start;
	DataType pivot = a[mid];

	for(size_t i=mid+1; i<=end; i++)
	{
		if(a[i] < pivot)
			swap(&a[++m], &a[i]);

		// sure that mid <= i
	}
	
	swap(&pivot, &a[mid]);
	
	// here a[i] < pivot when i < mid
	// a[i] >= pivot when i > mid

	// careful that size_t will overflow
	if (start+1 < mid)
		alg1(a, start, m-1);
	
	if(end-1 < mid)
		alg1(a, m+1, end);
}


int main()
{



	return 0;
}


