#include    "unp.h"

// for thundering herd problem, refer to section 30.6
void child_main(int i, int listenfd, int addrlen)
{
    void            web_child(int);

    struct sockaddr* cliaddr = Malloc(addrlen);
    printf("child %ld starting\n", (long)getpid());
    for ( ; ; )
    {
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
