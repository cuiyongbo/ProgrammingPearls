#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TRIALS	5
#define LOOPCOUNT	5000

#define T(s) printf("%s (trial = %d)\n", s, TRIALS)

#define M(op)	{	\
	printf("%-22s", #op);	\
	clock_t timeSum = 0;	\
	for (int ex=0; ex<TRIALS; ++ex) {	\
		k = 0;	\
		clock_t start = clock();	\
		for(i=1; i<=LOOPCOUNT; ++i) {	\
			fi = (float)i;	\
			for(j=1; j<=LOOPCOUNT; ++j) { 	\
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



int main()
{
	int i = 0;
	int j = 0;
	int k = 0;

	float fi = 0;
	float fj = 0;
	float fk = 0;

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
	M(fk = fi + fj);
	M(fk = fi - fj);
	M(fk = fi * fj);
	M(fk = fi / fj);

	return 0;
}
