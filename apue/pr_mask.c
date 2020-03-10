#include "apue.h"

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
