#include "apue.h"

void sigHandler(int signo)
{
    printf("%s signal received\n", strsignal(signo));
}

int main()
{
    if(signal(SIGUSR1, sigHandler) == SIG_ERR)
    {
        err_sys("signal error");
    }

    pid_t pid = fork();
    if(pid < 0)
    {
        err_sys("fork error");
    }
    else if(pid == 0)
    {
	printf("child: %d\n", (int)getpid());
        while(1)
        {
            sleep(50);
        }
        exit(0);
    }
    else
    {
        if(waitpid(pid, NULL, WNOHANG)<0)
        {
            err_sys("waitpid error");
        }
    }
    return 0;
}

