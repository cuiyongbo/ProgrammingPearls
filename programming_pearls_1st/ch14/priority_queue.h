#pragma once

template<typename T>
class PriorityQueue
{
public:
	PriorityQueue(int capacity);
	~PriorityQueue();
	void insert(T t);
	T extractMin();

private:
	void swap(int i, int j) {T t=m_data[i];m_data[i]=m_data[j];m_data[j]=t;}

private:
	int m_size;
	int m_capacity;
	T* m_data;
};

