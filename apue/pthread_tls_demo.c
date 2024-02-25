#include "apue.h"

#define  NUMTHREADS     3
static pthread_key_t   tlsKey = 0;
static pthread_once_t initOnce = PTHREAD_ONCE_INIT;

void globalDestructor(void *value)
{
  	printf("In the globalDestructor\n");
  	free(value);
  	pthread_setspecific(tlsKey, NULL);
}

void showGlobal(void)
{
  	pthread_t* tid = (pthread_t*)pthread_getspecific(tlsKey);
  	printf("showGlobal: global data stored for thread %#lx\n", *tid);
}

void initKey()
{
	printf("Create a thread local storage key\n");
  	int rc = pthread_key_create(&tlsKey, globalDestructor);
  	if(rc != 0)
  	{
		err_sys("pthread_key_create error");
  	}
}

void* threadfunc(void *parm)
{
  	printf("Inside secondary thread\n");
	pthread_once(&initOnce, initKey);
  	pthread_t me = pthread_self();
  	int* myThreadDataStructure = (int*)malloc(sizeof(pthread_t) + sizeof(int) * 10);
  	memcpy(myThreadDataStructure, &me, sizeof(pthread_t));
  	pthread_setspecific(tlsKey, myThreadDataStructure);
  	showGlobal();
  	pthread_exit(NULL);
}

int main(int argc, char **argv)
{
	int rc = 0;
  	printf("Create %d threads using joinable attributes\n", NUMTHREADS);
  	pthread_t thread[NUMTHREADS];
  	for (int i=0; i<NUMTHREADS; ++i)
  	{
  	  	rc = pthread_create(&thread[i], NULL, threadfunc, NULL);
  	  	if(rc != 0)
  	  	{
  	  		err_sys("pthread_create error");
  	  	}
  	}
	
  	printf("Join to threads\n");
  	for (int i=0; i<NUMTHREADS; ++i)
  	{
  	  	rc = pthread_join(thread[i], NULL);
  	  	if(rc != 0)
  	  	{
  	  		err_sys("pthread_join error");
  	  	}
  	}
	
  	printf("Delete a thread local storage key\n");
  	rc = pthread_key_delete(tlsKey);
  	if(rc != 0)
  	{
  		err_sys("pthread_key_delete error");
  	}
  	printf("Main completed\n");
  	
  	return 0;
}