#include "apue.h"

typedef void (*SigFunc)(int);

SigFunc* signal(int signo, SigFunc* func)
{
	struct sigaction act, oact;
	act.sa_handler = func;
	sigempty(&act.sa_mask);
	act.sa_flags = 0;
	if(signo == SIGALRM) {
#ifdef SA_INTERRUPT
	act.sa_flags |= SA_INTERRUPT;
#endif
	} else {
		act.sa_flags |= SA_RESTART;
	}

	if(sigaction(signo, &act, &oact) < 0)
		return SIG_ERR;
	return oact.sa_handler;
}

