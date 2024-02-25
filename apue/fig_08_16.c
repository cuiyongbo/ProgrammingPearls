#include "apue.h"

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("Usage: %s proc arg1 arg2 ...\n", argv[0]);
        return 1;
    }

	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork error");
	}
	else if(pid == 0)
	{
		if(execv(argv[1], argv+1)<0)
		{
			err_sys("execv error");
		}
	}
	else
	{
		printf("waiting for process %d\n", pid);
		if (waitpid(pid, NULL, WNOHANG) < 0)
		{
			err_sys("waitpid error");
		}
		exit(0);
	}
}

