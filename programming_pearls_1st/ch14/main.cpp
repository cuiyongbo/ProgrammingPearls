#include <stdio.h>
#include "priority_queue.cpp"

template class PriorityQueue<int>;
 
template<typename T>
void pqsort(T* v, int length)
{
	PriorityQueue<T> pq(length);
	for(int i=0; i<length; i++)
		pq.insert(v[i]);
	for(int i=0; i<length; i++)
		v[i] = pq.extractMin();
}

int main()
{
	const int n = 10;
	int v[n];
	for(int i=0; i<n; i++)
		v[i] = 10-i;

	pqsort<int>(v, n);

	for(int i=0; i<n; i++)
		printf("%d ", v[i]);
	printf("\n");

	return 0;
}

