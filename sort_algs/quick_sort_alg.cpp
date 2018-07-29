#include <stdio.h>

#define swapWithType(Type, a, b) {if(a != b) {Type tmp = a; a=b; b=tmp;}}
#define element_of(a) sizeof(a)/sizeof(a[0])

int partition(int* a, int lo, int hi)
{
	int pivot = a[hi];
	int i=lo-1;
	for(int j=lo; j<hi; j++)
	{
		if(a[j] < pivot)
		{
			i++;
			swapWithType(int, a[i], a[j]);
		}
	}
	swapWithType(int, a[i+1], a[hi]);
	return i+1;
}

void quickSort(int* a, int lo, int hi)
{
	if(lo >= hi)
		return;

	int p = partition(a, lo, hi);
	quickSort(a, lo, p-1);
	quickSort(a, p+1, hi);
}

int main()
{
	int a[] = {3, 7, 8, 5 ,2, 1, 9, 5, 4}; 

	quickSort(a, 0, element_of(a)-1);
	
	for(int i=0; i<element_of(a); i++)
		printf("%d ", a[i]);
	printf("\n");

	return 0;
}

