#include "apue.h"

void sigHandler(int signo)
{
    printf("PID<%d>: %s received\n", (int)getpid(), strsignal(signo));
    sleep(1);
}

int main()
{
    printf("main pid: %d\n", (int)getpid());
    if(signal(SIGSEGV, sigHandler) == SIG_ERR)
    {
        err_sys("signal error");
    }

    int* s = (int*)0;
    *s = 9;
}
