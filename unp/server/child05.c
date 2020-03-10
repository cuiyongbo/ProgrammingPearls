#include    "unp.h"
#include    "child.h"

void child_main(int i, int listenfd, int addrlen)
{
    char            c;
    int             connfd;
    void            web_child(int);

    printf("child %ld starting\n", (long) getpid());
    for ( ; ; )
    {
        if (Read_fd(STDERR_FILENO, &c, 1, &connfd) == 0)
            err_quit("read_fd returned 0");

        if (connfd < 0)
            err_quit("no descriptor from read_fd");

        web_child(connfd);              /* process request */
        Close(connfd);

        Write(STDERR_FILENO, "", 1);    /* tell parent we're ready again */
    }
}

pid_t child_make(int i, int listenfd, int addrlen)
{
    int sockfd[2];
    Socketpair(AF_LOCAL, SOCK_STREAM, 0, sockfd);

    pid_t pid = Fork();
    if (pid > 0)
    { // parent
        Close(sockfd[1]);
        g_cptr[i].child_pid = pid;
        g_cptr[i].child_pipefd = sockfd[0];
        g_cptr[i].child_status = 0;
        return pid;
    }

    Dup2(sockfd[1], STDERR_FILENO);     /* child's stream pipe to parent */
    Close(sockfd[0]);
    Close(sockfd[1]);
    Close(listenfd);                    /* child does not need this open */

    child_main(i, listenfd, addrlen);   /* never returns */
    return 0; // to avoid compilation warning
}
