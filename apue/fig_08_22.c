#include "apue.h"
#include <sys/wait.h>

int system(const char* cmdString)
{
	if (cmdString == NULL)
		exit(1);

	int status;
	pid_t pid = fork();
	if(pid < 0)
	{
		status = -1;
	}
	else if(pid == 0)
	{
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
	return status;
}

