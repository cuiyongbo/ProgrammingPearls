#include "apue.h"

void sig_alarm(int signo)
{
    printf("%s caught\n", strsignal(signo));
}

int main()
{
    if (signal(SIGALRM, sig_alarm) == SIG_ERR) 
    {
        err_sys("signal error");
    }

    size_t max_size = 1024*1024*1024;
    void* buff = malloc(max_size);
    if (buff == NULL)
    {
        err_sys("malloc error");
    }

    size_t rbytes = fread(buff, 1, max_size, stdin);
    if (ferror(stdin) != 0)
    {
        err_sys("fread error");
    }

    alarm(1);

    size_t wbytes = fwrite(buff, 1, rbytes, stdout);
    if (ferror(stdout) != 0)
    {
        err_sys("fwrite error");
    }

    return 0;
}