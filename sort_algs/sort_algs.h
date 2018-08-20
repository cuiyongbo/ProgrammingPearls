#pragma once

#define swapWithType(Type, a, b) {Type tmp=a; a=b; b= tmp;}

// Cautious, exchange does not work when a, b are the same variable
#define exchange(a, b) {a=a+b; b=a-b; a=a-b;}

#define local_min(a, b) ((a)<(b) ? (a) : (b))
#define local_max(a, b) ((a)>(b) ? (a) : (b))

void insertionSort(int* a, int count);
void insertionSort_noSwap(int* a, int count);

void mergeSort_topDown(int* a, int count);
void mergeSort_bottomUp(int* a, int count);

void heapSort_siftUp(int* a, int count);
void heapSort_siftDown(int* a, int count);

// Both l and r are inclusive
void quickSort(int* a, int l, int r);
void quickSort_hoare(int* a, int l, int r);
void quickSort_threeWayPartition(int* a, int l, int r);
