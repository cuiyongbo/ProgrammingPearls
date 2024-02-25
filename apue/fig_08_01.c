#include "apue.h"

static void childNotification(int signo)
{
    printf("a child has terminated\n");
    fflush(stderr);
    fflush(stdout);
}

int main()
{
    pid_t pid = fork();
    if(pid < 0)
    {
        err_sys("fork error");
    }
    else if(pid > 0)
    {
        if (signal(SIGCHLD, childNotification) == SIG_ERR)
        {
            err_sys("signal error");
        }
        wait(NULL);
    }
    else
    {
        printf("child: %ld\n", (long)getpid());
        while(1)
        {
            sleep(5);
        }
    }
    return 0;
}
