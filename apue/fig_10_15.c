#include "apue.h"
#include <signal.h>

static void sig_quit(int signo)
{
	printf("SIGQUIT caught\n");
	if(signal(SIGQUIT, SIG_DFL) == SIG_ERR)
		err_sys("signal(SIGQUIT)");
}

int main()
{
	if(signal(SIGQUIT, sig_quit) == SIG_ERR)
		err_sys("signal(SIGQUIT)");
	
	sigset_t newMask, oldMask, pendingMask;
	sigemptyset(&newMask);
	sigaddset(&newMask, SIGQUIT);
	if(sigprocmask(SIG_BLOCK, &newMask, &oldMask) < 0)
		err_sys("sigprocmask");

	sleep(5);

	if(sigpending(&pendingMask) < 0)
		err_sys("sigpending");
	if(sigismember(&pendingMask, SIGQUIT))
		printf("\nSIGQUIT pending\n");

	if(sigprocmask(SIG_SETMASK, &oldMask, NULL) < 0)
		err_sys("sigprocmask");

	sleep(5);
	exit(0);
}

