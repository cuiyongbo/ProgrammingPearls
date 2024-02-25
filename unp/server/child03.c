#include    "unp.h"

static void child_main(int i, int listenfd, int addrlen)
{
    void web_child(int);
    void my_lock_wait();
    void my_lock_release();

    struct sockaddr* cliaddr = Malloc(addrlen);
    printf("child %ld starting\n", (long)getpid());
    for ( ; ; )
    {
        socklen_t clilen = addrlen;

        // using file lock to avoid thundering herd problem
        my_lock_wait();
        int connfd = Accept(listenfd, cliaddr, &clilen);
        my_lock_release();

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
