#include "apue.h"

int main()
{
    pid_t pid = fork();
    if (pid < 0)
    {
        err_sys("fork error");
    }
    else if (pid > 0)
    {
        int status;
        pid_t ret_pid = waitpid(pid, &status, WUNTRACED);
        if (ret_pid < 0)
        {
            err_sys("waitpid error");
        }
        else
        {
            printf("child <%u> exit: ", (unsigned int)ret_pid);
            pr_exit(status);
        }

        if (open("/dev/tty", O_RDWR) < 0)
        {
            printf("parent: %u doesn't have a controlling terminal!\n", getpid());
        }
        else
        {
            printf("parent: %u has a controlling terminal!\n", getpid());
        }
    }
    else
    {
        printf("before setsid, process group ID: %u, ppid: %u, pid: %u\n", getpgrp(), getppid(), getpid());
        pid_t session_id = setsid();
        if (session_id < 0)
        {
            err_sys("setsid error");
        }

        printf("after setsid, process group ID: %u, pid: %u\n", getpgrp(), getpid());

        if (open("/dev/tty", O_RDWR) < 0)
        {
            printf("child: %u doesn't have a controlling terminal!\n", getpid());
        }
        else
        {
            printf("child: %u has a controlling terminal!\n", getpid());
        }
    }

    return 0;
}