#include    "unp.h"

// for select Collisions problem, refer to 30.6
static void child_main(int i, int listenfd, int addrlen)
{
    void            web_child(int);

    fd_set rset;
    FD_ZERO(&rset);
    struct sockaddr* cliaddr = Malloc(addrlen);
    printf("child %ld starting\n", (long)getpid());
    for ( ; ; )
    {
        FD_SET(listenfd, &rset);
        Select(listenfd+1, &rset, NULL, NULL, NULL);
        if(FD_ISSET(listenfd, &rset) == 0)
            err_quit("listenfd readable");

        socklen_t clilen = addrlen;
        int connfd = Accept(listenfd, cliaddr, &clilen);
        web_child(connfd);      /* process the request */
        Close(connfd);
    }
}

pid_t child_make(int i, int listenfd, int addrlen)
{
    pid_t   pid;
    if ((pid = Fork()) > 0)
        return pid;        /* parent */

    child_main(i, listenfd, addrlen);   /* never returns */
    return 0;
}
