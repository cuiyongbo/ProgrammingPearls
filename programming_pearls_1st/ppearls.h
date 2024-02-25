#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#if !defined(WIN32)
#include <unistd.h>
#endif

#ifdef _MSC_VER
#define forceinline __forceinline
#elif defined(__GNUC__)
#define forceinline __attribute__((always_inline)) inline
#else
#define forceinline inline
#endif

#define MULT 31
#define NHASH 29989

#define element_of_array(a) (sizeof(a)/sizeof(a[0]))
#define swapWithType(a, b, T) {T t=a;a=b;b=t;}
#define max(a, b) ((a)>(b)?(a):(b))
#define min(a, b) ((a)<(b)?(a):(b))

forceinline int isSorted(int* a, int length) {
	for (int i=1; i<length; i++) {
		if (a[i-1] > a[i]) {
			return 0;
		}
	}
	return 1;
}

forceinline void printArray(int* a, int length) {
	for (int i=0; i<length; i++) {
		printf("%d ", a[i]);
	}
	printf("\n");
}

forceinline int bigrand() { return RAND_MAX * rand() + rand();}
forceinline int randint(int l, int u) { return l + bigrand() % (u-l+1);} // problem: what if bigrand return a negative number

forceinline void genRandomArray(int* a, int length) {
 	srand((unsigned)time(NULL));
	for (int i=0; i<length; i++) {
		a[i] = (long)bigrand();
	}
}
