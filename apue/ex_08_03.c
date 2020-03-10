 #include "apue.h"

void printExitStatus(siginfo_t* infop);

int main(void)
{
    pid_t pid = fork();
    if (pid < 0)
        err_sys("fork error");
    else if (pid == 0)
        exit(7);

    siginfo_t infop;
    //int options = WCONTINUED | WEXITED | WNOHANG | WNOWAIT | WSTOPPED;
    int options = WCONTINUED | WEXITED | WNOWAIT | WSTOPPED;

    infop.si_pid = 0;
    if(waitid(P_PID, (id_t)pid, &infop, options) < 0)
        err_sys("waitid error");

    printExitStatus(&infop);

    if ((pid = fork()) < 0)
        err_sys("fork error");
    else if (pid == 0)
        abort();

    infop.si_pid = 0;
    if(waitid(P_PID, (id_t)pid, &infop, options) < 0)
        err_sys("waitid error");

    printExitStatus(&infop);

    if ((pid = fork()) < 0)
        err_sys("fork error");
    else if (pid == 0)
        kill(getpid(), SIGFPE);

    infop.si_pid = 0;
    if(waitid(P_PID, (id_t)pid, &infop, options) < 0)
        err_sys("waitid error");

    printExitStatus(&infop);
    
    exit(0); 
}

void printExitStatus(siginfo_t* infop)
{
    if (infop == NULL || infop->si_pid == 0)
        return;

    printf("Child %u: ", (unsigned int)infop->si_pid);

    switch(infop->si_code)
    {
        case CLD_EXITED:
        {
            printf("normal exit, status: %d\n", infop->si_status);
            break;
        }
        case CLD_KILLED:
        {
            printf("killed by signal: %d\n", infop->si_status);
            break;
        }
        case CLD_DUMPED:
        {
            printf("killed by signal: %d, core dumped\n", infop->si_status);
            break;
        }
        case CLD_STOPPED:
        {
            printf("stopped by signal: %d\n", infop->si_status);
            break;
        }
        case CLD_TRAPPED:
        {
            printf("trapped by signal: %d\n", infop->si_status);
            break;
        }
        case CLD_CONTINUED:
        {
            printf("contined: %d\n", infop->si_status);
            break;
        }
    }
}