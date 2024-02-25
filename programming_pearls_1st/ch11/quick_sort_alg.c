#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline void swap(int* l, int* r) {
	int t = *l;
	*l = *r;
	*r = t;
}

// a[start, end]
void quick_sort(int* a, int start, int end) {
	if (start >= end) {
		return;
	}

	int pivot = a[end];
	int m = start;
	for (int i=start; i<end; ++i) {
		if (a[i] < pivot) {
			swap(a+m, a+i);
			m++;
		}
	}

	// pivot is in its final sorted position
	swap(a+m, a+end);

	quick_sort(a, start, m-1);
	quick_sort(a, m+1, end);

/*
a s e m pivot
invariant:
1. for k in [s, m-1] a[k] < pivot
2. for k in [m, i-1] a[k] >= pivot

initialition: 
m = s-1, i=s
[s, m-1] is empty 
[m, i-1] is empty 
invariants stand

loop :
	if a[i] < pivot, swap(a[m], a[i]), m = m+1
	if a[i] >= pivot, do nothing
	invariant 1, 2 stand

end: 
	i = end
	[s, m-1] a[k] < pivot
	[m, end-1] a[k] >= pivot
*/
}

void print_array(int* a, int n) {
	for (int i=0; i<n; ++i) {
		printf("%d ", a[i]);
	}
	printf("\n");
}

void is_sorted(int* a, int n) {
	for (int i=0; i<n-1; ++i) {
		if (a[i] > a[i+1]) {
			printf("array is not sorted\n");
			//print_array(a, n);
			break;
		}
	}	
}

void scaffold() {
	int num = rand() % 10000;
	for (int i=0; i<num; ++i) {
		const int n = rand() % 10000;
		int* a = (int*)malloc(n * sizeof(int));
		for (int i=0; i<n; ++i) {
			a[i] = rand();
		}

		//print_array(a, n);
		quick_sort(a, 0, n-1);
		is_sorted(a, n);
		//print_array(a, n);

		free(a);
	}
}

int main() {
	srand((unsigned)time(NULL));
	scaffold();
	return 0;
}
