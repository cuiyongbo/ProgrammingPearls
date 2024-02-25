#include "apue.h"

#define THREAD_NUMBER	8
#define TOTAL_NUMBER 8000000L
#define NUMBER_PER_THREAD (TOTAL_NUMBER/THREAD_NUMBER)

long nums[TOTAL_NUMBER];
long snums[TOTAL_NUMBER];

pthread_barrier_t b;

#define heapsort qsort

//extern int heapsort(void*, size_t, size_t, int(*)(const void*, const void*));

int longCmp(const void* l, const void* r)
{
	long a = *(long*)l;
	long b = *(long*)r;
	return (a>b) - (a<b);
}

void* threadFunc(void* arg)
{
	long idx = (long)arg;
	heapsort(nums+idx, NUMBER_PER_THREAD, sizeof(long), longCmp);
	pthread_barrier_wait(&b);
	pthread_exit(NULL);
}

void merge()
{
	long idx[THREAD_NUMBER];
	for(long i=0; i<THREAD_NUMBER; i++)
		idx[i] = i*NUMBER_PER_THREAD;

	long num, minidx;
	for(long sidx=0; sidx<TOTAL_NUMBER; sidx++)
	{
		num = LONG_MAX;
		for(long i=0; i<THREAD_NUMBER; i++)
		{
			if((idx[i] < (i+1)*NUMBER_PER_THREAD) && (nums[idx[i]] < num))
			{
				num = nums[idx[i]];
				minidx = i;
			}	
		}

		snums[sidx] = nums[idx[minidx]];
		idx[minidx]++;
	}
}

int isSorted()
{
	for(unsigned long i=1; i<TOTAL_NUMBER; i++)
	{
		if(snums[i] < snums[i-1])
			return 0;
	}
	return 1;
}

int main()
{
	srandom((unsigned)time(NULL));
	for(unsigned long i=0; i<TOTAL_NUMBER; i++)
		nums[i] = random();

	struct timeval start, end;
	gettimeofday(&start, NULL);
	pthread_barrier_init(&b, NULL, THREAD_NUMBER+1);
	int err;
	pthread_t tid;
	for(int i=0; i<THREAD_NUMBER; i++)
	{
		err = pthread_create(&tid, NULL, threadFunc, (void*)(i*NUMBER_PER_THREAD));
		if(err != 0)
			err_exit(err, "pthread_create error");
	}

	pthread_barrier_wait(&b);
	merge();
	assert(isSorted());
	
	gettimeofday(&end, NULL);
	long long startuec, enduec;
	startuec = start.tv_sec*1000000 + start.tv_usec;
	enduec = end.tv_sec*1000000 + end.tv_usec;
	double elapsed = (enduec - startuec)/1000000.0;
	printf("sort took %.4f seconds\n", elapsed);

	return 0;
}


