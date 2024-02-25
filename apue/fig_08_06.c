 #include "apue.h"

int main(void)
{
    pid_t pid = fork();
    if (pid < 0)
        err_sys("fork error");
    else if (pid == 0)
        exit(7);

    int status;
    if (wait(&status) != pid)
        err_sys("wait error");

    pr_exit(status);

    if ((pid = fork()) < 0)
        err_sys("fork error");
    else if (pid == 0)
        abort();

    if (wait(&status) != pid)
        err_sys("wait error");

    pr_exit(status);

    if ((pid = fork()) < 0)
        err_sys("fork error");
    else if (pid == 0)
        status /= 0;

    if (wait(&status) != pid)
        err_sys("wait error");
    
    pr_exit(status);

    exit(0); 
}