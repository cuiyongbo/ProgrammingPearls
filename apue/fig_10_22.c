#include "apue.h"
#include <signal.h>

void pr_mask(const char *str)
{
	int errno_save = errno; /* we can be called by signal handlers */

	sigset_t sigset;
	if (sigprocmask(0, NULL, &sigset) < 0) 
		err_ret("sigprocmask error");

	printf("%s", str);
	if (sigismember(&sigset, SIGINT))
		printf(" SIGINT");
	if (sigismember(&sigset, SIGQUIT))
		printf(" SIGQUIT");
	if (sigismember(&sigset, SIGUSR1))
		printf(" SIGUSR1");
	if (sigismember(&sigset, SIGALRM))
		printf(" SIGALRM");
	/* remaining signals can go here */

	printf("\n");
	errno = errno_save; /* restore errno */
}

static void sig_int(int signo)
{
	pr_mask("in sig_int: ");
}

int main()
{
	pr_mask("program start: ");

	if(signal(SIGINT, sig_int) == SIG_ERR)
		err_sys("signal(SIGINT)");
	
	sigset_t newMask, oldMask;	
	sigemptyset(&newMask);
	sigaddset(&newMask, SIGINT);
	if(sigprocmask(SIG_BLOCK, &newMask, &oldMask) < 0)
		err_sys("sigprocmask(SIG_BLOCK)");

	pr_mask("in critical region: ");
	
	sigset_t waitMask;
	sigemptyset(&waitMask);
	sigaddset(&waitMask, SIGUSR1);
	if(sigsuspend(&waitMask) != -1)
		err_sys("sigsuspend");

	pr_mask("after return from sigsuspend: ");

	if(sigprocmask(SIG_SETMASK, &oldMask, NULL) < 0)
		err_sys("sigprocmask(SIG_SETMASK)");

	pr_mask("program exit: ");
	exit(0);
}

