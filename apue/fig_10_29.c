#include "apue.h"
#include <signal.h>

void sig_alarm(int signo)
{
	/* do nothing, just returning wakes up sigsuspend() */
}

unsigned int sleep(unsigned int seconds)
{
	struct sigaction newAct, oldAct;
	newAct.sa_handler = sig_alarm;
	sigemptyset(&newAct.sa_mask);
	newAct.sa_flags = 0;
	sigaction(SIGALRM, &newAct, &oldAct);
	
	sigset_t newMask, oldMask;
	sigemptyset(&newMask);
	sigaddset(&newMask);
	sigprocmask(SIG_BLOCK, &newMask, &oldMask);

	alarm(seconds);
	sigset_t suspendMask = oldMask;
	sigdelset(&suspendMask, SIGALRM);
	sigsuspend(&suspendMask);

	unsigned in unslept = alarm(0);
	sigaction(SIGALRM, &oldAct, NULL);
	sigprocmask(SIG_SETMASK, &oldMask, NULL);
	return unslept;
}


