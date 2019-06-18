#include "apue.h"

void sigHandler(int signo)
{
    printf("%s received\n", strsignal(signo));
    sleep(1);
}

int main()
{
    if(signal(SIGSEGV, sigHandler) == SIG_ERR)
    {
        err_sys("signal error");
    }

    int* s = (int*)0;
    *s = 9;
}
