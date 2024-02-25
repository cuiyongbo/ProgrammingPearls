#include <stdio.h>

void reverse(int* arr, int start, int end)
{
	for(; start < end; start++, end--)
	{
		int t = arr[start]; arr[start] = arr[end]; arr[end] = t;
	}
}

void rotate(int* arr, int length, int pivot)
{
	reverse(arr, 0, pivot-1);
	reverse(arr, pivot, length-1);
	reverse(arr, 0, length-1);
}

void printArray(int* arr, int length, const char* msg)
{
	printf("%s:\n", msg);
	for(int i=0; i<length; i++) printf("%d ", arr[i]);
	printf("\n");
}

int main()
{
	const int n = 10;
	int a[n];
	for(int i=0; i<n; i++) a[i] = i;

	printArray(a, n, "Before rotate(3)");
	rotate(a, n, 3);
	printArray(a, n, "After rotate(3)");

	return 0;
}

