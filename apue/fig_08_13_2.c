#include "apue.h"
#include <signal.h>

static volatile sig_atomic_t sigFlag;
static sigset_t newMask, oldMask, zeroMask;

static void sig_usr(int signo)
{
	sigFlag = 1;
}

void TELL_WAIT()
{
	if(signal(SIGUSR1, sig_usr) == SIG_ERR)
		err_sys("signal(SIGUSR1)");
	if(signal(SIGUSR2, sig_usr) == SIG_ERR)
		err_sys("signal(SIGUSR2)");

	sigemptyset(&zeroMask);
	sigemptyset(&newMask);
	sigaddset(&newMask, SIGUSR1);
	sigaddset(&newMask, SIGUSR2);
	if(sigprocmask(SIG_BLOCK, &newMask, &oldMask) < 0)
		err_sys("sigprocmask(SIG_BLOCK)");
}

void TELL_PARENT(pid_t pid)
{
	kill(pid, SIGUSR2);
}

void WAIT_PARENT()
{
	while(sigFlag == 0)
		sigsuspend(&zeroMask);
	sigFlag = 0;
	if(sigprocmask(SIG_SETMASK, &oldMask, NULL) < 0)
		err_sys("sigprocmask(SIG_SETMASK)");
}

void TELL_CHILD(pid_t pid)
{
	kill(pid, SIGUSR1);
}

void WAIT_CHILD()
{
	while(sigFlag == 0)
		sigsuspend(&zeroMask);
	sigFlag = 0;

	if(sigprocmask(SIG_SETMASK, &oldMask, NULL) < 0)
		err_sys("SIG_SETMASK");
}

static void charatatime(char* str)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	int c;
	for(char* ptr=str; (c=*ptr) != 0; ptr++)
		putc(c, stdout);
}

int main()
{
	TELL_WAIT();
	
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork");
	}
	else if(pid == 0)
	{
		WAIT_PARENT();
		charatatime("output from child\n");
	}
	else
	{
		charatatime("output from parent\n");
		TELL_CHILD(pid);
	}
	exit(0);
}

