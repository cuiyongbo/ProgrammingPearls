#include "apue.h"
#include <signal.h>
#include <sys/wait.h>

int system_local(const char* cmdString)
{
	if(cmdString == NULL)
		return 1;

	struct sigaction ignore, saveintr, savequit;
	ignore.sa_handler = SIG_IGN;
	sigemptyset(&ignore.sa_mask);
	ignore.sa_flags = 0;

	if(sigaction(SIGINT, &ignore, &saveintr) < 0)
		return -1;
	if(sigaction(SIGQUIT, &ignore, &savequit) < 0)
		return -1;

	sigset_t chldMask, saveMask;
	sigemptyset(&chldMask);
	sigaddset(&chldMask, SIGCHLD);
	if(sigprocmask(SIG_BLOCK, &chldMask, &saveMask) < 0)
		return -1;

	int status;
	pid_t pid = fork();
	if(pid < 0)
	{
		status = -1;
	}
	else if(pid == 0)
	{
		sigaction(SIGINT, &saveintr, NULL);
		sigaction(SIGQUIT, &savequit, NULL);
		sigprocmask(SIG_SETMASK, &saveMask, NULL);

		execl("/bin/sh", "sh", "-c", cmdString, (char*)0);
		_exit(127);
	}
	else
	{
		while(waitpid(pid, &status, 0) < 0)
		{
			if(errno != EINTR)
			{
				status = -1;
				break;
			}
		}
	}

	if(sigaction(SIGINT, &saveintr, NULL) < 0)
		return -1;
	if(sigaction(SIGQUIT, &savequit, NULL) < 0)
		return -1;

	if(sigprocmask(SIG_SETMASK, &saveMask, NULL) < 0)
		return -1;

	return status;
}

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
	if(system_local("/bin/ed") < 0)
		err_sys("system_local");
	exit(0);
}


