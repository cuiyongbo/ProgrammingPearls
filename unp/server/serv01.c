#include    "unp.h"

void sig_child(int);
void web_child(int);
void pr_cpu_time(void);

void sig_int(int signo)
{
    pr_cpu_time();
    exit(0);
}

int main(int argc, char **argv)
{
    int listenfd = 0;
    socklen_t addrlen = 0;
    if (argc == 2)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else if (argc == 3)
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);
    else
        err_quit("usage: %s [ <host> ] <port#>", argv[0]);

    struct sockaddr* cliaddr = Malloc(addrlen);

    Signal(SIGCHLD, sig_child);
    Signal(SIGINT, sig_int);

    for ( ; ; )
    {
        socklen_t clilen = addrlen;
        int connfd = accept(listenfd, cliaddr, &clilen);
        if (connfd < 0) {
            if (errno == EINTR)
                continue;       /* back to for() */
            else
                err_sys("accept error");
        }

        pid_t childpid = Fork();
        if (childpid == 0)
        {
            Close(listenfd);    /* close listening socket */
            web_child(connfd);  /* process request */
            exit(0);
        }
        Close(connfd);          /* parent closes connected socket */
    }
}
