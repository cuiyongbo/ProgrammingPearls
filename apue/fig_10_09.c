#include <setjmp.h>
#include <signal.h>
#include "apue.h"

static jmp_buf env_alarm;

static void sig_alarm(int signo)
{
	longjmp(env_alarm, 1);
}

unsigned int sleep2(unsigned int seconds)
{
	if (signal(SIGALRM, sig_alarm) == SIG_ERR)
		return seconds;	

	if(setjmp(env_alarm) == 0)
	{
		alarm(seconds);
		pause();
	}
	return alarm(0);
}

static void sig_int(int signo)
{
	volatile int k = 0;
	printf("\nsig_int starting\n");
	for(int i=0; i<300000; i++)
		for(int j=0; j<4000; j++)
			k += i*j;
	printf("sig_int finished\n");
}

int main()
{
	if(signal(SIGINT, sig_int) == SIG_ERR)
	{
		perror("signal");
		exit(EXIT_FAILURE);
	}

	unsigned int unslept = sleep2(5);
	printf("sleep2 returned: %u\n", unslept);
	exit(0);
}

