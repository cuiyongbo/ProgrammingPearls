#include "apue.h"

int quitflag;
sigset_t mask;

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t waitloc = PTHREAD_COND_INITIALIZER;

void* threadfunc(void* arg)
{
	int err, signo;
	for(;;)
	{
		err = sigwait(&mask, &signo);
		if(err != 0)
			err_exit(err, "sigwait failed");
		switch(signo)
		{
		case SIGINT:
			printf("\ninterrupted\n");
			break;
		case SIGQUIT:
			pthread_mutex_lock(&lock);
			quitflag = 1;
			pthread_mutex_unlock(&lock);
			pthread_cond_signal(&waitloc);
			return 0;
		default:
			printf("unexpected signal %d\n", signo);
			break;
		}
	}
}

int main()
{
	sigset_t oldmask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGINT);
	sigaddset(&mask, SIGQUIT);
	int err = pthread_sigmask(SIG_BLOCK, &mask, &oldmask);
	if(err != 0)
		err_exit(err, "pthread_sigmask error");

	pthread_t tid;	
	err = pthread_create(&tid, NULL, threadfunc, 0);
	if(err != 0)
		err_exit(err, "pthread_create error");

	quitflag = 0;
	pthread_mutex_lock(&lock);
	while(quitflag == 0)
		pthread_cond_wait(&waitloc, &lock);
	pthread_mutex_unlock(&lock);
	
	quitflag = 0;
	if(sigprocmask(SIG_SETMASK, &oldmask, NULL) < 0)
		err_sys("sigprocmask error");

	return 0;
}

