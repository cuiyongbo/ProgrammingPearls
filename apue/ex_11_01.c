#include "apue.h"

//typedef void* (*thread_func_t)(void*);

typedef struct
{
	int a, b, c, d;
} Foo;

void printFoo(const char* s, const Foo* fp)
{
	printf("%s:\n", s);
	printf("\tstructure at %#lx\n", (unsigned long)fp);
	printf("\tfoo.a = %d\n", fp->a);
	printf("\tfoo.b = %d\n", fp->b);
	printf("\tfoo.c = %d\n", fp->c);
	printf("\tfoo.d = %d\n", fp->d);
}

void* threadFunc1(void* arg)
{
	Foo* foo = (Foo*)malloc(sizeof(Foo)); 
	foo->a = 1;
	foo->b = 2;
	foo->c = 3;
	foo->d = 4;
	printFoo("thread 1", foo);
	pthread_exit((void*)foo);
} 

void* threadFunc2(void* arg)
{
	printf("thread 2 ID: %#lx\n", (unsigned long)pthread_self());
	pthread_exit((void*)0);
} 

int main()
{
	int err;
	pthread_t tids[2];
	thread_func_t* funcs[2] = {threadFunc1, threadFunc2};
	for(int i=0; i<2; i++)
	{
		err = pthread_create(tids+i, NULL, funcs[i], NULL);
		if(err != 0)
			err_exit(err, "pthread_create failed to create thread %d", i+1);
	}

	Foo* fp;
	err = pthread_join(tids[0], (void*)&fp);
	if(err != 0)
		err_exit(err, "pthread_join failed to join thread 1");
	printFoo("parent", fp);	
	free(fp);
	sleep(1);

	void* tret;
	err = pthread_join(tids[1], &tret);
	if(err != 0)
		err_exit(err, "pthread_join failed to join thread 2");
	printf("thread 2 exit code %ld\n", (long)tret);
	sleep(1);

	return 0;
}
