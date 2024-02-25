/*
Polynomial to evalute:
	y = a[n]*x^n + a[n-1]*x^(n-1) + ... + a[1]*x + a[0]
*/

#include <stdio.h>

/*
a - coefficient array
n - size of coefficient array , which meams the order of polynomial is n-1
*/


/* alg1 will perform 2*n multiplifications*/
int alg1(int* a , int n, int x)
{
	int y = a[0];
	int xi = 1;
	for(int i=1; i<n; i++)
	{
		xi *= x;
		y += a[i]*xi;
	}

	return y;
}

/* alg2 will perform n multiplifications*/
int alg2(int* a , int n, int x)
{
	int y = a[n-1];

	for(int i=n-2; i>=0; i--)
		y = x*y + a[i];

	return y;
}

