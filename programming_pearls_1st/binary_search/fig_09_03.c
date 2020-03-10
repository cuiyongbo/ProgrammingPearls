#include "ppearls.h"

int binarySearch(int* arr, int length, int key)
{
	int l = 0;
	int u = length-1;
	while(l <= u)
	{
		int m = l + (u-l)/2;
		if(arr[m] == key)
			return m;
		else if(arr[m] < key)
			l = m + 1;
		else
			u = m - 1;
	}
	return -1;
}

int binarySearchWithLowerBound(int* a, int length, int key)
{
	int l = -1;
	int u = length;
	while(l+1 != u)
	{ /*invariant: a[l] < key && a[u] >= key && l < u*/
		int m = l + (u-l)/2;
		if(a[m] < key)
			l = m;
		else
			u = m;
	}

	/*assert l+1=u && x[l] < key && x[u] >= key*/
	if(u>=length || a[u] != key)
		u = -1;
	return u;
}

static int x[1000];
int binarySearchWithLowerBound_2(int k)
{
	int i = 512;	
	int l = -1;

	/*
		x[511] < k 
			probe 488, 488+1, 488+1+2, ..., 999[488+1+2+4+...+256]
		else
			probe 0, 1, 2, ... , 511[1+2+3+ ... + 256] 
	*/
	if(x[511] < k)
		l = 1000 - 512;
	while(i != 1)
	{ /*invariant: x[l]<k && x[l+i]>=k && i = 2^j*/
		int nexti = i >> 1;
		if(x[l+nexti] < k)
		{
			l += nexti;
			i = nexti;
		}
		else
			i = nexti;		
	}
	/* assert i==1 && x[l]<k && x[l+i]>=k */
	int p = l+1;
	if(p >= 1000 || x[p] != k)
		p = -1;
	return p;
}

int binarySearchWithLowerBound_3(int k)
{
	int i = 512;	
	int l = -1;
	if(x[511] < k)
		l = 1000 - 512;
	while(i != 1)
	{ /*invariant: x[l]<k && x[l+i]>=k && i = 2^j*/
		i >>= 1;	
		if(x[l+i] < k)
			l += i;
	}
	/* assert i==1 && x[l]<k && x[l+i]>=k */
	int p = l+1;
	if(p >= 1000 || x[p] != k)
		p = -1;
	return p;
}

int binarySearchWithLowerBound_4(int t)
{
	int l = -1;
	if(x[511] < t) l = 1000-512;
	if(x[l+256] < t) l += 256;
	if(x[l+128] < t) l += 128;
	if(x[l+64] < t) l += 64;
	if(x[l+32] < t) l += 32;
	if(x[l+16] < t) l += 16;
	if(x[l+8] < t) l += 8;
	if(x[l+4] < t) l += 4;
	if(x[l+2] < t) l += 2;
	int p = l+1;
	if(p >= 1000 || x[p] != t)
		p = -1;
	return p;
}

// find the leftmost 1-bit
int hibit2(int n)
{
   n |= (n >> 1);
   n |= (n >> 2);
   n |= (n >> 4);
   n |= (n >> 8);
   n |= (n >> 16);

   return n - (n >> 1);
}

int binarySearchWithLowerBound_5(int* a, int length, int k)
{
	int i = hibit2(length);
	int l = -1;
	if(a[i-1] < k)
		l = length - i;
	while(i != 1)
	{ /*invariant: a[l]<k && a[l+i]>=k && i = 2^j*/
		i >>= 1;	
		if(a[l+i] < k)
			l += i;
	}
	/* assert i==1 && a[l]<k && a[l+i]>=k */
	int p = l+1;
	if(p >= length || a[p] != k)
		p = -1;
	return p;
}

int main()
{
	int n = 1000;
	int* a = (int*)malloc(n*sizeof(int));
	for(int i=0; i<n; i++)
		a[i] = i>>2;
	printf("a[n] = %d\n", a[1000]);
	int res;
	res = binarySearchWithLowerBound_5(a, n, 5);
	assert(res == 20);
	res = binarySearchWithLowerBound_5(a, n, 250);
	assert(res == -1);
	free(a);
}



