#include "apue.h"

int makeThread(thread_func_t func, void* arg)
{
	int err;
	pthread_t tid;
	pthread_attr_t attr;
	err = pthread_attr_init(&attr);
	if(err != 0)
		return err;
	err = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	if(err != 0)
		return err;
	err = pthread_create(&tid, &attr, func, arg);
	pthread_attr_destroy(&attr);
	return err;	
}

