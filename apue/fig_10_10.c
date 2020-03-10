#include "apue.h"
#include <signal.h>

static void sig_alarm(int signo) {}

int main()
{
	if(signal(SIGALRM, sig_alarm) == SIG_ERR)
	{
		perror("signal(SIGALRM) failed");
		exit(EXIT_FAILURE);
	}

	alarm(10);
	char line[MAXLINE];
	int n = read(STDIN_FILENO, line, MAXLINE);
	if(n < 0)
	{
		perror("read");
		exit(EXIT_FAILURE);
	}
	alarm(0);
	write(STDOUT_FILENO, line, n);
	exit(0);
}

