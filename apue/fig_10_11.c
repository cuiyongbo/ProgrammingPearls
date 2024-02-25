#include "apue.h"
#include <signal.h>
#include <setjmp.h>

static jmp_buf env_alarm;

static void sig_alarm(int signo) 
{
	longjmp(env_alarm, 1);
}

int main()
{
	if(signal(SIGALRM, sig_alarm) == SIG_ERR)
	{
		err_sys("signal(SIGALRM) failed");
	}

	if(setjmp(env_alarm) != 0)
	{
		fprintf(stderr, "read timeout\n");
		exit(EXIT_FAILURE);
	}

	alarm(10);
	char line[MAXLINE];
	int n = read(STDIN_FILENO, line, MAXLINE);
	if(n < 0)
	{
		err_sys("read error");
	}
	alarm(0);
	write(STDOUT_FILENO, line, n);
	exit(0);
}

