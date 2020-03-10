#include "priority_queue.h"

template<typename T>
PriorityQueue<T>::PriorityQueue(int capacity)
{
	m_size = 0;
	m_capacity = capacity;
	m_data = new T[capacity+1];
}

template<typename T>
PriorityQueue<T>::~PriorityQueue()
{
	delete[] m_data;
}

template<typename T>
void PriorityQueue<T>::insert(T t)
{
	m_data[++m_size] = t;
	int i, p;
	// extend heap(1, m_size-1) to heap(1, m_size) by siftUp
	for(i=m_size; i>1&&m_data[p=i/2]>m_data[i]; i=p)
		swap(i, p);
}

template<typename T>
T PriorityQueue<T>::extractMin()
{
	T t = m_data[1];
	m_data[1] = m_data[m_size--];
	int i, c;
	// extend heap(2, m_size) to heap(1, m_size) by siftDown
	for(i=1; (c=2*i)<=m_size; i=c)
	{
		if(c+1<=m_size && m_data[c+1] < m_data[c])
			c++;
		if(m_data[i] <= m_data[c])
			break;
		swap(i, c);
	}
	return t;
}

