#include    "unp.h"

int my_open(const char *pathname, int mode)
{
    int sockfd[2];
    Socketpair(AF_LOCAL, SOCK_STREAM, 0, sockfd);

    pid_t childpid = Fork();
    if (childpid == 0)
    { /* child process */
        Close(sockfd[0]);

        char argsockfd[10], argmode[10];
        snprintf(argsockfd, sizeof(argsockfd), "%d", sockfd[1]);
        snprintf(argmode, sizeof(argmode), "%d", mode);
        execl("./openfile", "openfile", argsockfd, pathname, argmode, (char *) NULL);
    }

    /* parent process - wait for the child to terminate */
    Close(sockfd[1]); /* close the end we don't use */

    int fd, status;
    Waitpid(childpid, &status, 0);
    if (WIFEXITED(status) == 0)
        err_quit("child did not terminate");

    if ((status = WEXITSTATUS(status)) == 0)
    {
        char c;
        Read_fd(sockfd[0], &c, 1, &fd);
    }
    else
    {
        fd = -1;
        errno = status;     /* set errno value from child's status */
    }

    Close(sockfd[0]);
    return fd;
}
