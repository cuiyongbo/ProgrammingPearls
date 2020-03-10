#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TRIALS	5
#define LOOPCOUNT  5000

#define T(s) printf("%s (trial = %d)\n", s, TRIALS)

#define M(op)	{	\
	printf("%-30s", #op);	\
	clock_t timeSum = 0;	\
	for (int ex=0; ex<TRIALS; ++ex) {	\
		k = 0;	\
		clock_t start = clock();	\
		for(int i=1; i<=loops; ++i) {	\
			fi = (float)i;	\
			for(int j=1; j<=loops; ++j) { 	\
				fj = (float)j;	\
				op;	\
			}	\
		}	\
		clock_t during = clock() - start;	\
		printf("%6d ", (int)during);	\
		timeSum += during;	\
	}	\
	float avg = 1e09 * timeSum/((double)LOOPCOUNT*LOOPCOUNT*TRIALS*CLOCKS_PER_SEC);	\
	printf("%8.0f\n", avg);	\
}

#define MAXN	100000
int x[MAXN];

int intcmp(int *i, int *j) { return *i - *j;}

#define swapmac(i, j) {int t=x[i]; x[i]=x[j]; x[j]=t;}
void swapfunc(int i, int j)
{
	int t = x[i];
	x[i] = x[j];
	x[j] = t;
}

#define maxmac(a, b) ((a)>(b)?(a):(b))
int maxfunc(int a, int b) { return a>b?a:b; }


int main()
{
	int k = 0;
	int loops = LOOPCOUNT;
	float fi, fj, fk;
	fi = fj =  fk = 0;

	T("Integer Algorithm");
	M({});
	M(k++);
	M(++k);
	M(k = i + j);
	M(k = i - j);
	M(k = i * j);
	M(k = i / j);
	M(k = i % j);
	M(k = i & j);
	M(k = i | j);

	T("Float Algorithm");
	M(fj = j);
	M(fk = fi + fj);
	M(fk = fi - fj);
	M(fk = fi * fj);
	M(fk = fi / fj);

	T("Array Operations");
	for(int i=0; i<LOOPCOUNT; ++i)
		x[i] = rand();

	M(k = i + j);
	M(k = x[i] + j;);
	M(k = i + x[j];);
	M(k = x[i] + x[j];);
	
	T("Comparisons");
	M(if(i < j) k++);
	M(if(x[i] < x[j]) k++);

	T("Array Comparisons and Swaps");	
	M(k = (x[i]<x[j]) ? -1 : 1);
	M(k = intcmp(x+i, x+j));
	M(swapfunc(i, j));
	M(swapmac(i, j));
	
	T("Max Function, Macro, and Inline");
	M(k = (i>j)? i : j);
	M(k = maxmac(i, j));
	M(k = maxfunc(i, j));

	loops = LOOPCOUNT/5;

	T("Math Functions");
	M(k = rand());
	M(fk = j + fi);
	M(fk = sqrt(j+fi));
	M(fk = sin(j+fi));
	M(fk = sinh(j+fi));
	M(fk = asin(j+fi));
	M(fk = cos(j+fi));
	M(fk = tan(j+fi));

	loops = LOOPCOUNT/100;

	T("Memory Allocation");
	M(free(malloc(16)));
	M(free(malloc(100)));
	M(free(malloc(2000)));

	return 0;
}
