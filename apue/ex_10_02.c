#include "apue.h"

extern const char *const sys_siglist[_NSIG];

int sig2str(int signo,  char* str)
{
    if(signo < 0 || signo > NSIG || str == NULL)
    {
        return -1;
    }

    // It is up to the caller to ensure there is enough
    // space in str, including the null terminal byte.
    strcpy(str, sys_siglist[signo]);
}

int main()
{
    char buff[32];
    int signo = 3;

    printf("Enter a signal number (no larger than %d): ", NSIG);
    while(scanf("%d", &signo) != EOF)
    {
        if(sig2str(signo, buff) == -1)
            break;
        
        printf("Signal number %d: %s\n", signo, buff);
        printf("Enter a signal number: ");
    }
}
