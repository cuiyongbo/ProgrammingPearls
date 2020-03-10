/* include serv05a */
#include    "unp.h"
#include    "child.h"

static int g_nchildren;

int main(int argc, char **argv)
{
    void        sig_int(int);
    pid_t       child_make(int, int, int);

    int listenfd = 0;
    socklen_t addrlen = 0;
    if (argc == 3)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else if (argc == 4)
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);
    else
        err_quit("usage: serv05 [ <host> ] <port#> <#children>");

    fd_set rset, masterset;
    FD_ZERO(&masterset);
    FD_SET(listenfd, &masterset);

    g_nchildren = atoi(argv[argc-1]);
    g_cptr = Calloc(g_nchildren, sizeof(Child));

    int maxfd = listenfd;
    for (int i = 0; i < g_nchildren; i++)
    {
        child_make(i, listenfd, addrlen); // parent returns
        FD_SET(g_cptr[i].child_pipefd, &masterset);
        maxfd = max(maxfd, g_cptr[i].child_pipefd);
    }

    Signal(SIGINT, sig_int);

    int navail = g_nchildren;
    struct sockaddr* cliaddr = (struct sockaddr*)Malloc(addrlen);
    for ( ; ; )
    {
        rset = masterset;
        if (navail <= 0)
            FD_CLR(listenfd, &rset);    /* turn off if no available children */

        int nsel = Select(maxfd + 1, &rset, NULL, NULL, NULL);

        if (FD_ISSET(listenfd, &rset))
        {
            socklen_t clilen = addrlen;
            int connfd = Accept(listenfd, cliaddr, &clilen);

            int i = 0;
            for (; i < g_nchildren; i++)
            {
                if (g_cptr[i].child_status == 0)
                    break;
            }

            if (i == g_nchildren)
                err_quit("no available children");

            g_cptr[i].child_status = 1;   /* mark child as busy */
            g_cptr[i].child_count++;
            navail--;

            Write_fd(g_cptr[i].child_pipefd, "", 1, connfd);
            Close(connfd);

            if (--nsel == 0)
                continue;
        }

        for (int i = 0; i < g_nchildren; i++)
        {
            if (FD_ISSET(g_cptr[i].child_pipefd, &rset))
            {
                int rc;
                if (Read(g_cptr[i].child_pipefd, &rc, 1) == 0)
                    err_quit("child %d terminated unexpectedly", i);

                g_cptr[i].child_status = 0;
                navail++;

                if (--nsel == 0)
                    break;
            }
        }
    }
}

void sig_int(int signo)
{
    void    pr_cpu_time(void);

    for (int i = 0; i < g_nchildren; i++)
        kill(g_cptr[i].child_pid, SIGTERM);

    while (wait(NULL) > 0)      /* wait for all children */
        ;

    if (errno != ECHILD)
        err_sys("wait error");

    pr_cpu_time();

    for (int i = 0; i < g_nchildren; i++)
        printf("child %d, %ld connections\n", i, g_cptr[i].child_count);

    exit(0);
}
