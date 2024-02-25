#include <stdio.h>

#define swapWithType(a, b, T) {T t=a;a=b;b=t;}

int randint(int l, int u)
{ // todo: return a random integer between l and u
	return (l+u)/2;
}

// find the kth element, zero-based.
void select_nthElement(int* a, int l, int u, int k)
{
	if(l >= u)
		return;

	swapWithType(a[l], a[randint(l, u)], int);
	int m = l;
	for(int i=l+1; i<=u; i++)
	{
		if(a[i] < a[l])
		{
			++m;
			swapWithType(a[i], a[m], int);
		}
	}
	swapWithType(a[m], a[l], int);
	if(m>k)
		select_nthElement(a, l, m-1, k);
	else if(m<k)
		select_nthElement(a, m+1, u, k);
}


// find the kth element, one-based.
void select_nthElement_2(int* a, int l, int u, int k)
{
	if(l >= u)
		return;

	swapWithType(a[l], a[randint(l, u)], int);
	int m = l;
	for(int i=l+1; i<=u; i++)
	{
		if(a[i] < a[l])
		{
			++m;
			swapWithType(a[i], a[m], int);
		}
	}
	swapWithType(a[m], a[l], int);
	int q = m-l+1;
	if(q == k)
		return;
	if(q > k)
		select_nthElement_2(a, l, m-1, k);
	else if(q < k)
		select_nthElement_2(a, m+1, u, k-q);
}

void select_nthElement_loop(int* a, int l, int u, int k)
{
	while(l < u)
	{
		swapWithType(a[l], a[randint(l, u)], int);
		int m = l;
		for(int i=l+1; i<=u; i++)
		{
			if(a[i] < a[l])
			{
				++m;
				swapWithType(a[i], a[m], int);
			}
		}
		swapWithType(a[m], a[l], int);
		if(m>k)
			u = m-1;
		else if(m<k)
			l = m+1;
		else
			break;
	}
}

int main()
{
	int a[] = {1, 2, 4, 1, 5, 7, 3, 9};
	int length = sizeof(a)/sizeof(a[0]);
	//select_nthElement(a, 0, length-1, length/2);
	select_nthElement_2(a, 0, length-1, length/2);
	//select_nthElement_loop(a, 0, length-1, length-1);
	for(int i=0; i<length; i++)
		printf("%d ", a[i]);
	printf("\n");
	return 0;
}


