#include "apue.h"
#define BUFFSIZE 100

void sig_intr(int signo)
{
    printf("%s caught\n", strsignal(signo));
}

int main(void)
{
    if(signal(SIGXFSZ, sig_intr) == SIG_ERR)
    {
        err_sys("signal error");
    }

    // struct rlimit {
    //     rlim_t rlim_cur;  /* Soft limit */
    //     rlim_t rlim_max;  /* Hard limit (ceiling for rlim_cur) */
    // };
    
    struct rlimit rlim;
    if (getrlimit(RLIMIT_FSIZE, &rlim) != 0)
    {
        err_sys("getrlimit error");
    }

    printf("Soft limit: %d, hard limit: %d\n", (int)rlim.rlim_cur, (int)rlim.rlim_max);

    rlim.rlim_cur = 1024;
    if (setrlimit(RLIMIT_FSIZE, &rlim) != 0)
    {
        err_sys("setrlimit error");
    }

    int rbytes;
    char buf[BUFFSIZE];
    while ((rbytes = read(STDIN_FILENO, buf, BUFFSIZE)) > 0)
    {
        int wbytes = write(STDOUT_FILENO, buf, rbytes);
        if (wbytes != rbytes)
        {
            printf("read bytes: %d, written bytes: %d\n", rbytes, wbytes);
        }
    }
    
    if (rbytes < 0)
        err_sys("read error");

    strcpy(buf, "Hello world!\n");
    if (write(STDOUT_FILENO, buf, strlen(buf)) < 0)
    {
        err_sys("write error");
    }
    
    exit(0);
}