#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <assert.h>

#define MAXN	1000000

typedef int DataType;

DataType x[MAXN];
int n;

//int i = -999999;
#define assert(v) {if((v) == 0) printf(" binarysearch bug %d %d\n", i, n);}

int binarySearch1(DataType t)
{
	int m;
	int l = 0;
	int u = n-1;
	for(;;) {
		if (l > u)
			return -1;
		m = (l+u)/2;
		if(x[m]<t)
			l = m+1;
		else if (x[m] == t)
			return m;
		else /* x[m] > t */
			return u = m-1;
	}
}

int binarySearch2(DataType t)
{
	int l, u, m;
	l = 0;
	u = n-1;
	while(l <= u) {
		m = (l+u)/2;
		if(x[m] < t)
			l = m+1;
		else if(x[m]==t)
			return m;
		else /* x[m] > t */
			u = m-1;
	}
	return -1;
}

int binarySearch3(DataType t)
{
	int l, u, m;
	l = -1;
	u = n;
	while(l+1 != u) {
		m = (l+u)/2;
		if(x[m] < t)
			l = m;
		else 
			u = m;
	}
	if(u>=n || x[u] != t)
		return -1;
	return u;
}

int seqSearch1(DataType t)
{
	int i;
	for(i=0; i<n; i++)
		if(x[i] == t)
			return i;
	return -1;
}

// faster than version 1 because it skips if control in for
int seqSearch2(DataType t)
{
	int i = 0;
	DataType hold = x[n];
	x[n] = t;
	for(i=0;;i++)
		if (x[i] == t)
			break;
	x[n] = hold;
	if(i==n)
		return -1;
	else
		return i;
}

DataType p[MAXN];

// reshuffle p[]
void scramble(int n)
{
	int i;
	long long j;
	DataType t;
	for(i=n-1; i>0; i--) {
		j = ((long long)RAND_MAX*rand() + rand()) % (i+1); // [0, i]
		t=p[i];p[i]=p[j];p[j]=t;
	}
}

void printArray(DataType* arr, int n)
{
	int i;
	for(i=0; i< n; i++)
		printf("%d%s", arr[i], (i%10==9 ? "\n" : "\t"));
}

void timeDriver()
{
	srand(time(NULL));
	int i, algnum, numtests, test, start, clicks;
	while (scanf("%d %d %d", &algnum, &n, &numtests) != EOF) {
		for (i=0; i<n; i++)	
			x[i] = i; p[i] = i;

		scramble(n);
		printArray(p, n);

		start = clock();
		for(test=0; test<numtests; test++) {
			for(i=0; i<n; i++) {
				switch(algnum) {
				case 1: assert(binarySearch1(p[i]) == p[i]); break;					
				case 2: assert(binarySearch2(p[i]) == p[i]); break;					
				case 3: assert(binarySearch3(p[i]) == p[i]); break;					
				//case 4: assert(binarySearch4(p[i]) == p[i]); break;					
				//case 9: assert(binarySearch9(p[i]) == p[i]); break;					
				case 21: assert(seqSearch1(p[i]) == p[i]); break;					
				case 22: assert(seqSearch2(p[i]) == p[i]); break;					
				//case 23: assert(seqSearch3(p[i]) == p[i]); break;					
				}
			}	
		}
		clicks = clock() - start;
		printf("%d\t%d\t%d\t%d\t%g\n", 
				algnum, n, numtests, clicks,
				1e09*clicks/((float) CLOCKS_PER_SEC*n*numtests));
	}
}

int main()
{
	timeDriver();
	return 0;
}

