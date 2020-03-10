#include "apue.h"
#include <setjmp.h>
static jmp_buf env;

void sigHandler(int signo)
{
    printf("%s received\n", strsignal(signo));
    longjmp(env, 1);
}

int main()
{
    if(signal(SIGSEGV, sigHandler) == SIG_ERR)
    {
        err_sys("signal error");
    }

    if(setjmp(env) == 0)
    {
        int* s = (int*)0;
        *s = 9;
    }
    else
    {
        printf("after sigHandler\n");
    }
}
