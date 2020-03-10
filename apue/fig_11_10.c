#include "apue.h"

typedef struct
{
	int f_id;
	int f_count;
	pthread_mutex_t f_lock;
	// other members ...
} Foo;

Foo* foo_alloc(int id)
{
	Foo* fp = (Foo*)malloc(sizeof(Foo));
	if(fp == NULL)
		err_sys("malloc error");

	fp->f_count = 1;
	fp->f_id = id;
	int err = pthread_mutex_init(&fp->f_lock, NULL);
	if(err != 0)
	{
		free(fp);
		fp = NULL;
		err_ret("pthread_mutex_init error: %s", strerror(err));
	}	
	return fp;
}

void foo_hold(Foo* fp)
{
	pthread_mutex_lock(&fp->f_lock);
	fp->f_count++;
	pthread_mutex_unlock(&fp->f_lock);
}

void foo_rele(Foo* fp)
{
	pthread_mutex_lock(&fp->f_lock);
	if(--fp->f_count == 0)
	{ // what if another thread is block waiting foo_hold?
		pthread_mutex_unlock(&fp->f_lock);
		pthread_mutex_destroy(&fp->f_lock);
		free(fp);
	}
	else
	{
		pthread_mutex_unlock(&fp->f_lock);
	}	
}

