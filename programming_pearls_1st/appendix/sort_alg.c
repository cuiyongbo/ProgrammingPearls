#include <stdio.h>

#define TRUE	1
#define FALSE	0

typedef int BOOL;
typedef int DataType;

void swap(DataType* l, DataType* r)
{
	DataType t = *l;
	*l = *r;
	*r = t;
}

void selectionSort(DataType* a, int n)
{
	for(int i=0; i<n-1; i++)
	{
		DataType minElement = a[i];
		int minElementIdx = i;
		for(int j=i+1; j<n; j++)
		{
			if(minElement > a[j]) 
			{
				minElement = a[j];
				minElementIdx = j;
			}
		}

		if(i != minElementIdx)
			swap(&a[i], &a[minElementIdx]);
	}
}

void bubbleSort(DataType* a, int n)
{
	for(int i=0; i<n-1; i++)
	{
		BOOL isAlreadyOrdered = TRUE; 

		for(int j=0; j<n-i-1; j++)
		{
			if(a[j]>a[j+1])
			{
				swap(&a[j], &a[j+1]);
				isAlreadyOrdered = FALSE;
			}	
		}

		if(isAlreadyOrdered)
			break;
	}
}

int main(void)
{

	return 0;
}

