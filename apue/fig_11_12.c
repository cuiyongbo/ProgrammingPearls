#include "apue.h"

#define NHASH 29
#define HASH(id) (((unsigned long)id)%NHASH)

typedef struct Foo
{
	int f_id;
	int f_count; // protected by hashLock
	pthread_mutex_t f_lock;
	struct Foo* f_next; // protected by hashLock
	// other members ...
} Foo;

Foo* fh[NHASH];
pthread_mutex_t hashLock = PTHREAD_MUTEX_INITIALIZER;

Foo* foo_alloc(int id)
{
	Foo* fp = (Foo*)malloc(sizeof(Foo));
	if(fp == NULL)
	{
		err_msg("malloc error");
		return NULL;
	}

	int err = pthread_mutex_init(&fp->f_lock, NULL);
	if(err != 0)
	{
		free(fp);
		err_ret("pthread_mutex_init error: %s", strerror(err));
		return NULL;
	}	

	fp->f_id = id;
	fp->f_count = 1;
	int idx = HASH(id);
	pthread_mutex_lock(&hashLock);
	fp->f_next = fh[idx];
	fh[idx] = fp;
	pthread_mutex_lock(&fp->f_lock);
	pthread_mutex_unlock(&hashLock);
	// other initializations go here
	pthread_mutex_unlock(&fp->f_lock);
		
	return fp;
}

void foo_hold(Foo* fp)
{
	pthread_mutex_lock(&hashLock);
	fp->f_count++;
	pthread_mutex_unlock(&hashLock);
}

void foo_rele(Foo* fp)
{
	pthread_mutex_lock(&hashLock);
	if(--fp->f_count == 0)
	{ 
		Foo* tfp = fh[HASH(fp->f_id)];
		if(tfp == fp)
		{
			fh[idx] = fp->f_next;
		}
		else
		{
			while(tfp->f_next != fp)
				tfp = tfp->f_next;
			tfp->f_next = fp->f_next;
		}
		pthread_mutex_destroy(&fp->f_lock);
		free(fp);
	}
	pthread_mutex_unlock(&hashLock);
}

Foo* foo_find(int id)
{
	Foo* fp;
	pthread_mutex_lock(&hashLock);
	for(fp=fh[HASH(id)]; fp != NULL; fp=fp->f_next)
	{
		if(fp->f_id == id)
		{
			foo_hold(fp);
			break;
		}
	}
	pthread_mutex_unlock(&hashLock);
	return fp;
}

