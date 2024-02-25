#include "apue.h"
#include <signal.h>

static volatile sig_atomic_t quitFlag;

static void sig_handler(int signo)
{
	if(signo == SIGINT)
		printf("\ninterrupt\n");
	else if(signo == SIGQUIT)
		quitFlag = 1;
}

int main()
{
	if(signal(SIGINT, sig_handler) == SIG_ERR)
		err_sys("signal(SIGINT)");
	if(signal(SIGQUIT, sig_handler) == SIG_ERR)
		err_sys("signal(SIGQUIT)");

	sigset_t newMask, oldMask;
	sigemptyset(&newMask);
	sigaddset(&newMask, SIGQUIT);
	if(sigprocmask(SIG_BLOCK, &newMask, &oldMask) < 0)
		err_sys("sigprocmask(SIG_BLOCK)");

	sigset_t zeroMask;
	sigemptyset(&zeroMask);
	while(quitFlag == 0)
		sigsuspend(&zeroMask);

	quitFlag = 0;

	if(sigprocmask(SIG_SETMASK, &oldMask, NULL) < 0)
		err_sys("sigprocmask(SIG_SETMASK)");
	
	exit(0);
}

