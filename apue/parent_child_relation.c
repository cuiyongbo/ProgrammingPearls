#include "apue.h"

int failed_to_reap_all_children();
int succeeded_in_reaping_all_children();

int main()
{
    succeeded_in_reaping_all_children();
}

int succeeded_in_reaping_all_children()
{
    const int N = 3;
    pid_t childPIds[N];

    for (int i = 0; i < N; ++i)
    {
        pid_t pid = fork();
        if (pid < 0)
        {
            err_sys("fork error");
        }
        else if (pid == 0)
        {
            printf("Child: %u\n", getpid());
            exit(0);
        }
        else
        {
            childPIds[i] = pid;
        }
    }

    while(1)
    {
        int status;
        pid_t ret_pid = wait(&status);
        if (ret_pid < 0)
        {
            if (errno == ECHILD)
            {
                printf("All children reaped\n");
                break;
            }
            else if(errno == EINTR)
            {
                continue;
            }
            else
            {
                err_sys("wait error");
            }
        }
        else
        {
            printf("child<%u>: ", ret_pid);
            pr_exit(status);
        }
    }

    char cmdBuf[128] = "ps ";
    for (int i = 0; i < N; ++i)
    {
        sprintf(cmdBuf + strlen(cmdBuf), "%u,", childPIds[i]);
    }
    sprintf(cmdBuf + strlen(cmdBuf), "%d", 1);
    system(cmdBuf);

    return 0;
}

int failed_to_reap_all_children()
{
    const int N = 3;
    pid_t childPIds[N];

    for (int i = 0; i < N; ++i)
    {
        pid_t pid = fork();
        if (pid < 0)
        {
            err_sys("fork error");
        }
        else if (pid == 0)
        {
            printf("Child: %u\n", getpid());
            exit(0);
        }
        else
        {
            childPIds[i] = pid;
        }
    }

    int status;
    pid_t ret_pid = wait(&status);
    if (ret_pid < 0)
    {
        err_sys("wait error");
    }
    else
    {
        printf("child<%u>: ", ret_pid);
        pr_exit(status);
    }

    char cmdBuf[128] = "ps ";
    for (int i = 0; i < N; ++i)
    {
        sprintf(cmdBuf + strlen(cmdBuf), "%u,", childPIds[i]);
    }

    sprintf(cmdBuf + strlen(cmdBuf), "%d", 1);

    printf("CMD: %s\n", cmdBuf);
    system(cmdBuf);

    return 0;
}
