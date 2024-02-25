#include "apue.h"

#define SECTONSEC	1000000000

struct to_info
{
	thread_func_t* to_fn;
	void* to_arg;
	struct timespec to_wait;
};

void* timeout_helper(void* arg)
{
	struct to_info* tip = (struct to_info*)arg;
	nanosleep(&tip->to_wait, NULL);
	(*tip->to_fn)(tip->to_arg);
	free(arg);
	return (void*)0;
}

void timeout(const struct timespec* when, thread_func_t func, void* arg)
{
	int err;
	struct to_info* tip;
	struct timespec now;
	clock_gettime(CLOCK_REALTIME, &now);
	if((when->tv_sec > now.tv_sec)
		|| (when->tv_sec == now.tv_sec && when->tv_nsec > now.tv_nsec))
	{
		tip = (struct to_info*)malloc(sizeof(struct to_info));
		if(tip != NULL)
		{
			tip->to_fn = func;
			tip->to_arg = arg;
			tip->to_wait.tv_sec = when->tv_sec - now.tv_sec;
			if(when->tv_nsec >= now.tv_nsec)	
			{
				tip->to_wait.tv_nsec = when->tv_nsec - now.tv_nsec;
			}
			else
			{
				tip->to_wait.tv_sec--;
				tip->to_wait.tv_nsec = SECTONSEC - now.tv_nsec + when->tv_nsec;
			}
			err = makeThread(timeout_helper, (void*)tip);
			if(err == 0)
				return;
			else
				free(tip);
		}
	}

	// we get here if (a) when <= now, or (b) malloc fails,
	// (c) we cann't make a thread, so we just call the function now.
	(*func)(arg);
}

pthread_mutexattr_t mutexattr;
pthread_mutex_t mutex;

void* retry(void* arg)
{
	pthread_mutex_lock(&mutex);
	// perform retry steps ...
	pthread_mutex_unlock(&mutex);
	return NULL;
}

int main()
{
	int err, condition, arg;
	err = pthread_mutexattr_init(&mutexattr);
	if(err != 0)
		err_exit(err, "pthread_mutexattr_init");
	err = pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_RECURSIVE);
	if(err != 0)
		err_exit(err, "pthread_mutexattr_settype");
	err = pthread_mutex_init(&mutex, &mutexattr);
	if(err != 0)
		err_exit(err, "pthread_mutexattr_init");

	// some more initialization ...

	pthread_mutex_lock(&mutex);

	// check the condition under the protection of a lock
	// to make the check and the call to timeout atomic

	struct timespec when;
	if(condition)
	{
		// calculate the absolute time when we want to retry
		clock_gettime(CLOCK_REALTIME, &when);
		when.tv_sec += 10; 
		timeout(&when, retry, (void*)arg);
	}

	// continue processing

	pthread_mutex_unlock(&mutex);
	return 0;
}
