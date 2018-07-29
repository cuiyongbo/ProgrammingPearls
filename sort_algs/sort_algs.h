#pragma once

#define swapWithType(Type, a, b) {Type tmp=a; a=b; b= tmp;}
#define exchage(a, b) {a=a+b; b=a-b; a=a-b;}
#define local_min(a, b) ((a)<(b) ? (a) : (b))
#define local_max(a, b) ((a)>(b) ? (a) : (b))

void insertionSort(int* a, int count);
void insertionSort_noSwap(int* a, int count);

void mergeSort_topDown(int* a, int count);
void mergeSort_bottomUp(int* a, int count);

