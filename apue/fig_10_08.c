#include <setjmp.h>
#include <signal.h>
#include <unistd.h>

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

