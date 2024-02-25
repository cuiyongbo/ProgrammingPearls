#include "unp.h"

void sig_child(int signo)
{
    pid_t pid;
    while((pid=waitpid(-1, NULL, WNOHANG)) > 0)
        printf("child <%d> terminated\n", (int)pid);
}
