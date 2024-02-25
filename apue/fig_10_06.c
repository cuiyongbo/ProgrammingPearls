#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

static void sig_cld(int sigNo)
{
	printf("SIGCLD received.\n");
	if(signal(SIGCLD, sig_cld) == SIG_ERR)
	{
		perror("signal");
	}

	pid_t pid;
	int status;
	if((pid=wait(&status)) < 0)
		perror("wait");
	
	printf("pid: %ld, exit status: %d\n", (long)pid, status);
}

int main()
{
	if(signal(SIGCLD, sig_cld) == SIG_ERR)
	{
		perror("signal");
		exit(EXIT_FAILURE);
	}

	pid_t pid = fork();
	if(pid < 0)
	{
		perror("fork");
		exit(EXIT_FAILURE);
	}
	else if(pid == 0)
	{
		sleep(2);
		_exit(0);
	}

	pause();
	return 0;
}

