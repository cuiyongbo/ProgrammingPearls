#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

void err_sys(const char* msg)
{
	perror(msg);
	exit(EXIT_FAILURE);
}

int main()
{
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork");
	}
	else if(pid == 0)
	{
		if(execlp("testinterp", "testinterp", "arg1", (char*)0) < 0)
		//if(execl("./awkTest", "awkTest", "arg1", (char*)0) < 0)
			err_sys("execl");
	}

	if(waitpid(pid, NULL, 0) < 0)
		err_sys("waitpid");

	exit(EXIT_SUCCESS);
}

