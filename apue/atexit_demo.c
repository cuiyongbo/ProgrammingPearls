#include "apue.h"

void bye(void)
{
    printf("%ld: That was all, folks\n", (long)getpid());
	// exit(0);
}

void haha(void)
{
    printf("%ld: That was haha, folks\n", (long)getpid());
	_exit(0);
}

int main(void)
{
    long a = sysconf(_SC_ATEXIT_MAX);
    printf("ATEXIT_MAX = %ld\n", a);
	
    if (atexit(bye) != 0) 
    {
		err_sys("atexit error");
    }

    if (atexit(haha) != 0) 
    {
		err_sys("atexit error");
    }
	
/*  Functions registered using atexit() (and on_exit(3)) are not called if a process terminates abnormally because of the delivery of a signal. 
	while(1)
	{
		sleep(30);
	}
*/

/* case 1: child inherits copies of its parent's registrations
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork error");
	}
	else if(pid == 0)
	{
		exit(0);
	}
	else
	{
		if(waitpid(pid, NULL, WNOHANG) != 0)
		{
			err_sys("waitpid error");
		}
	}
*/
    exit(EXIT_SUCCESS);
}

