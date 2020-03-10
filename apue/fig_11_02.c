#include "apue.h"

void printids(const char* s)
{
	pid_t pid = getpid();
	pthread_t tid = pthread_self();
	printf("%s: pid %lu tid %lu(%#lx)\n",
		s, (unsigned long)pid, (unsigned long)tid, (unsigned long)tid);
}

void* threadFunc(void* arg)
{
	printids("new thread");
	pthread_exit(NULL);
}

int main()
{
	pthread_t tid;
	int err = pthread_create(&tid, NULL, threadFunc, NULL);
	if(err != 0)
		err_exit(err, "pthread_create error");
	printids("main thread");
	sleep(1);
	return 0;
}

