#include "apue.h"

void handler(int signo)
{
    printf("Signal %d <%s> receieved\n", signo, strsignal(signo));
    sleep(10);
    return;
}

int main()
{
    if(signal(SIGUSR1, handler) == SIG_ERR)
        err_sys("signal error");

    printf("start process %d\n", (int)getpid());
    while(1)
    {
        sleep(10);
    }
    return 0;
}

