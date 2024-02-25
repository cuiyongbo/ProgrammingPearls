#include "apue.h"
#include <signal.h>
#include <setjmp.h>

static sigjmp_buf jmpbuf;
static volatile sig_atomic_t canjump;

void pr_mask(const char* str);
static void sig_usr1(int signo);
static void sig_alarm(int signo);

int main()
{
	if(signal(SIGUSR1, sig_usr1) == SIG_ERR)
		err_sys("signal(SIGUSR1)");
	if(signal(SIGALRM, sig_alarm) == SIG_ERR)
		err_sys("signal(SIGALRM)");

	pr_mask("starting main: ");

	if(sigsetjmp(jmpbuf, 1) != 0)
	{
		pr_mask("ending main: ");
		exit(0);
	}
	canjump = 1;

	for(;;)
		pause();
}


void pr_mask(const char* str)
{
	int errno_save = errno;
	
	sigset_t sigset;
	if(sigprocmask(0, NULL, &sigset) < 0)
	{
		perror("sigprocmask");
		return;
	}

	printf("%s", str);
	if(sigismember(&sigset, SIGINT))
		printf(" SIGINT");	
	if(sigismember(&sigset, SIGQUIT))
		printf(" SIGQUIT");	
	if(sigismember(&sigset, SIGUSR1))
		printf(" SIGUSR1");	
	if(sigismember(&sigset, SIGALRM))
		printf(" SIGALRM");	
	/* remaining signals can go here */
	printf("\n");	

	errno = errno_save;
}

static void sig_usr1(int signo)
{
	if(canjump == 0)
		return;

	pr_mask("starting sig_usr1: ");
	
	alarm(3);	
	time_t startTime = time(NULL);
	for(;;)
	{
		if(time(NULL) > startTime + 5)
			break;
	}

	pr_mask("finishing sig_usr1: ");
	canjump = 0;
	siglongjmp(jmpbuf, 1);
}

static void sig_alarm(int signo)
{
	pr_mask("in sig_alarm: ");
}

