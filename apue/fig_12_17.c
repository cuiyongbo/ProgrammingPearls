#include "apue.h"

pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

void prepare()
{
	int err;
	printf("preparing locks...\n");
	err = pthread_mutex_lock(&lock1);
	if(err != 0)
		err_cont(err, "can't lock lock1 in prepare handler");
	err = pthread_mutex_lock(&lock2);
	if(err != 0)
		err_cont(err, "can't lock lock2 in prepare handler");
}

void parent()
{
	int err;
	printf("parent unlocking locks...\n");
	err = pthread_mutex_unlock(&lock1);
	if(err != 0)
		err_cont(err, "can't unlock lock1 in parent handler");
	err = pthread_mutex_unlock(&lock2);
	if(err != 0)
		err_cont(err, "can't unlock lock2 in parent handler");
}

void child()
{
	int err;
	printf("child unlocking locks...\n");
	err = pthread_mutex_unlock(&lock1);
	if(err != 0)
		err_cont(err, "can't unlock lock1 in child handler");
	err = pthread_mutex_unlock(&lock2);
	if(err != 0)
		err_cont(err, "can't unlock lock2 in child handler");
}

void* threadfunc(void* arg)
{
	printf("thread started...\n");
	pause();
	pthread_exit(0);
}

int main()
{
	int err;
	pthread_t tid;
	err = pthread_atfork(prepare, parent, child);
	if(err != 0)
		err_exit(err, "pthread_atfork error");
	err = pthread_create(&tid, NULL, threadfunc, NULL);
	if(err != 0)
		err_exit(err, "pthread_create error");
	
	sleep(2);
	printf("parent about to fork...\n");
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork error");
	}
	else if(pid == 0)
	{
		printf("child returned from fork\n");
	}
	else
	{
		printf("parent returned from fork\n");
	}
	return 0;
}
