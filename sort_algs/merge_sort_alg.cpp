#include "sort_algs.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// end is excluive;
static void _mergeRoutine(int* src, int begin, int mid, int end, int* dest)
{
	int i = begin;
	int j = mid;
	for(int k=begin; k<end; k++)
	{
		if(i<mid && (j>=end || src[i]<=src[j]))
			dest[k] = src[i++];
		else
			dest[k] = src[j++];
	}	
}

static void _topDownSplitMerge(int* src, int begin, int end, int* dest);
void mergeSort_topDown(int* a, int count)
{
	int* duplicate = duplicateArray(a, 0, count);		
	
	_topDownSplitMerge(duplicate, 0, count, a);	 

	freeDuplicateArray(duplicate);
}

// begin is inclusive, and end is exclusive.
static void _topDownSplitMerge(int* src, int begin, int end, int* dest)
{
	if(end - begin < 2)
		return;

	int mid = begin + (end-begin)/2;
	_topDownSplitMerge(dest, begin, mid, src);
	_topDownSplitMerge(dest, mid, end, src);

	_mergeRoutine(src, begin, mid, end, dest);	
}

void mergeSort_bottomUp(int* a, int count)
{
	int* duplicate = new int[count];
	
	for(int width=1; width<count; width *= 2)
	{
		for(int i=0; i<count; i += 2*width)
			_mergeRoutine(a, i, local_min(i+width, count), local_min(i+2*width, count), duplicate);		
		memmove(a, duplicate, sizeof(int)*count);
	}
	delete[] duplicate;
}

