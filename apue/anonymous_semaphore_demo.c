#include "apue.h"
#include "signal.h"
#include "semaphore.h"

sem_t g_sem;

static void handler(int signo)
{
	write(STDOUT_FILENO, "sem_post from handler\n", 24);
	if(sem_post(&g_sem) < 0)
	{
		write(STDERR_FILENO, "sem_post failed\n", 18);
		_exit(EXIT_FAILURE);
	}
}

int main(int argc, char* argv[])
{
	if(argc != 3)
		err_quit("Usage: %s <alarm-secs> <wait-secs>", argv[0]);

	if(sem_init(&g_sem, 0, 0) < 0)
		err_sys("sem_init error");
	
	struct sigaction sa;
	sa.sa_handler = handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	if(sigaction(SIGALRM, &sa, NULL) < 0)
		err_sys("sigaction(SIGALARM) error");

	alarm(atoi(argv[1]));

	struct timespec ts;
	if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
		err_sys("clock_gettime(CLOCK_REALTIME) error");

	ts.tv_sec += atoi(argv[2]);
	printf("main about to call sem_timedwait\n");
	
	int s;
	do
	{
		s = sem_timedwait(&g_sem, &ts);
	} while(s == -1 && errno == EINTR);

	if(s==-1)
	{
		if(errno == ETIMEDOUT)
			printf("sem_timedwait timeout\n");
		else
			perror("sem_timedwait error");
	}
	else
	{
		printf("sem_timedwait succeeded\n");
	}
	
	if(sem_destroy(&g_sem) < 0)
		err_sys("sem_destroy error");

	return s==0 ? EXIT_SUCCESS : EXIT_FAILURE;

}
