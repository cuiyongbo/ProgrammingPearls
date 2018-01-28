#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int DataType;

void swap(DataType* a, DataType *b)
{
	DataType t = *a;
	*a = *b;
	*b = t;
}


void alg1(DataType *arr, size_t arrSize)
{
	for(size_t i=1; i<arrSize; i++)
		for(size_t j=i; j>0&&arr[j-1]>arr[j]; j--)
			swap(&arr[j], &arr[j-1]);
}

void alg2(DataType* arr, size_t arrSize)
{
	for(size_t i=1; i<arrSize; i++) {
		DataType t = arr[i];
		size_t j=i;
		for(; j>0&&arr[j-1]>t; j--)
			arr[j] = arr[j-1];
		arr[j] = t;
	}
}


int main()
{


	return 0;
}

