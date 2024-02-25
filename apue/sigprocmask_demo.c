#include "apue.h"

void handler(int signo)
{
	printf("process<%d>: %s <%d> signal received\n", (int)getpid(), strsignal(signo), signo);
	sleep(5);
}

int main()
{
	printf("start process: %d\n", (int)getpid());
	struct sigaction nact, oact;
	nact.sa_handler = handler;
	sigemptyset(&nact.sa_mask);
	nact.sa_flags = SA_RESTART;

	if(sigaction(SIGUSR1, &nact, &oact) < 0)
		err_sys("sigaction error");

/*
	if(signal(SIGUSR1, handler) == SIG_ERR)
		err_sys("signal error");
*/

/*
	sigset_t nmask, omask;
	sigemptyset(&nmask);
	sigaddset(&nmask, SIGUSR1);
	if(sigprocmask(SIG_BLOCK, &nmask, &omask) < 0)
		err_sys("sigprocmask error");
*/

	while(1)
	{
		sleep(10);
	}

	return 0;
}
