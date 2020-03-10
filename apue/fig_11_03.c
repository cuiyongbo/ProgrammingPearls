#include "apue.h"

typedef void* (*thread_func_t)(void*);

void* threadFunc1(void* arg)
{
	printf("thread 1 returning\n");
	pthread_exit((void*)1);
} 

void* threadFunc2(void* arg)
{
	printf("thread 2 returning\n");
	pthread_exit((void*)2);
} 

int main()
{
	int err;
	pthread_t tids[2];
	thread_func_t funcs[2] = {threadFunc1, threadFunc2};
	for(int i=0; i<2; i++)
	{
		err = pthread_create(tids+i, NULL, funcs[i], NULL);
		if(err != 0)
			err_exit(err, "pthread_create failed to create thread %d", i+1);
	}

	void* tret;
	for(int i=0; i<2; i++)
	{
		err = pthread_join(tids[i], &tret);
		if(err != 0)
			err_exit(err, "pthread_join failed to join thread %d", i+1);
		printf("thread %d exit code %ld\n", i+1, (long)tret);
	}
	
	return 0;
}
