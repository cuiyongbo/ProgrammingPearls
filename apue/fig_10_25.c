#include "apue.h"
#include <signal.h>

/* POSIX-style abort() function */
void abort()
{
	struct sigaction action;	
	sigaction(SIGABRT, NULL, &action);
	if(action.sa_handler == SIG_IGN)
	{
		action.sa_handler = SIG_DFL;
		sigaction(SIGABRT, &action, NULL);
	}

	if(action.sa_handler == SIG_DFL)
		fflush(NULL);

	/* Caller can’t block SIGABRT; make sure it’s unblocked */
	sigset_t mask;
	sigfillset(&mask);
	sigdelset(&mask, SIGABRT);
	sigprocmask(SIG_SETMASK, &mask, NULL);
	kill(getpid(), SIGABRT);

	/* If we’re here, process caught SIGABRT and returned */
	fflush(NULL);
	action.sa_handler = SIG_DFL;
	sigaction(SIG_SETMASK, &mask, NULL);
	kill(getpid(), SIGABRT);
	exit(1); /* this should never be executed. */
}
