#include "apue.h"

void pr_mask_ex(const char* str)
{
    int errno_save = errno; 
    
    sigset_t sigset;
    if(sigprocmask(0, NULL, &sigset) < 0)
    {
        err_sys("sigprocmask error");
    }

    printf("%s: ", str);

    for (int i=0; i<NSIG; i++)
    {
        if (sigismember(&sigset, i))
            printf(" %s", strsignal(i));
    }

    printf("\n");   

    errno = errno_save;
}

int main()
{
    srand(time(NULL));

    sigset_t old_sigset, new_sigset;

    sigemptyset(&new_sigset);
    for (int i = 0; i < 5; ++i)
    {
        sigaddset(&new_sigset, rand()%NSIG);
    }

    if(sigprocmask(SIG_BLOCK, &new_sigset, &old_sigset) < 0)
    {
        err_sys("sigprocmask error");
    }

    pr_mask_ex("Current thread");


    sigprocmask(SIG_SETMASK, &old_sigset, NULL);
}