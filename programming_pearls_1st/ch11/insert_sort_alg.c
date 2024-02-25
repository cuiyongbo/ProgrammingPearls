#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printArray(int* arr, int length, const char* msg)
{
	printf("%s:\n", msg);
	for(int i=0; i<length; i++) printf("%d ", arr[i]);
	printf("\n");
}

void swap(int* a, int *b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

void alg1(int *arr, size_t arrSize)
{
	for(size_t i=1; i<arrSize; i++)
		for(size_t j=i; j>0&&arr[j-1]>arr[j]; j--)
			swap(arr+j, arr+j-1);
}

void alg2(int *arr, size_t arrSize)
{
	for(size_t i=1; i<arrSize; i++) 
	{
		for(size_t j=i; j>0&&arr[j-1]>arr[j]; j--) 
		{ 
			int t=arr[j-1];arr[j-1]=arr[j];arr[j]=t;
		}
	}
}

void alg3(int* arr, size_t arrSize)
{
	for(size_t i=1; i<arrSize; i++) 
	{
		size_t j=i;
		int t = arr[i];
		for(; j>0&&arr[j-1]>t; j--) arr[j] = arr[j-1];
		arr[j] = t;
	}
}

int main()
{
	const int n = 10;
	int a[n];
	for(int i=0; i<n; i++) a[i] = n-i; 

	printArray(a, n, "Before sort");

	//alg1(a, n);
	//alg2(a, n);
	alg3(a, n);

	printArray(a, n, "After sort");

	return 0;
}

