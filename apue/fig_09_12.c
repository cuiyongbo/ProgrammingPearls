#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

void sig_hup(int sig)
{
	printf("SIGHUP(%d) received, pid: %ld\n", sig, (long)getpid());
}

void pr_ids(const char* name)
{
	printf("%s: pid(%ld) ppid(%ld) pgrg(%ld) tpgrp(%ld)\n",
			name, (long)getpid(), (long)getppid(), 
			(long)getpgrp(), (long)tcgetpgrp(STDIN_FILENO));
	fflush(stdout);
}

void err_sys(const char* msg)
{
	perror(msg);
	exit(1);
}

int main()
{
	pr_ids("parents");
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork");
	}
	else if(pid > 0)
	{
		sleep(5);
	}
	else
	{
		pr_ids("child");
		signal(SIGHUP, sig_hup);
		kill(getpid(), SIGTSTP);
		pr_ids("child");
		char c;
		if(read(STDIN_FILENO, &c, 1) != 1)
			perror("read");
	}
	exit(0);
}
