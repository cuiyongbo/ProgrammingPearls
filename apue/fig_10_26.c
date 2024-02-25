#include "apue.h"
#include <signal.h>

static void sig_int(int signo)
{
	printf("caught SIGINT\n");
}

static void sig_child(int signo)
{
	printf("caught SIGCHLD\n");
}

int main()
{
	if(signal(SIGINT, sig_int) == SIG_ERR)
		err_sys("signal(SIGINT)");
	if(signal(SIGCHLD, sig_child) == SIG_ERR)
		err_sys("signal(SIGCHLD)");
	if(system("/bin/ed") < 0)
		err_sys("system");
	exit(0);
}
