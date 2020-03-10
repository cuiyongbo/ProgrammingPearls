#include "apue.h"
#include <signal.h>

static void sig_tstp(int signo)
{
	sigset_t mask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGTSTP);
	sigprocmask(SIG_UNBLOCK, &mask, NULL);

	signal(SIGTSTP, SIG_DFL);
	kill(getpid(), SIGTSTP);
	signal(SIGTSTP, sig_tstp);
}

int main()
{
	if(signal(SIGTSTP, SIG_IGN) == SIG_DFL)
		signal(SIGTSTP, sig_tstp);
	
	int n;
	char buf[BUFFSIZE];
	while((n = read(STDIN_FILENO, buf, BUFFSIZE)) > 0)
	{
		if(write(STDOUT_FILENO, buf, n) != n)
			err_sys("write error");
	}
	if(n < 0)
		err_sys("read error");
	exit(0);
}

