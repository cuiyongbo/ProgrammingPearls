#include    "unp.h"

static int      g_nchildren;
static pid_t*   g_pids;

int main(int argc, char **argv)
{
    void        sig_int(int);
    pid_t       child_make(int, int, int);

    int         listenfd = 0;
    socklen_t   addrlen = 0;
    if (argc == 3)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else if (argc == 4)
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);
    else
        err_quit("usage: serv02 [ <host> ] <port#> <#children>");

    g_nchildren = atoi(argv[argc-1]);
    g_pids = (pid_t*)Calloc(g_nchildren, sizeof(pid_t));

    for (int i = 0; i < g_nchildren; i++)
        g_pids[i] = child_make(i, listenfd, addrlen); /* parent returns */

    // The parent keeps the listening socket open in case
    // it needs to fork additional children at some later time,
    // which would be an enhancement to our code

    Signal(SIGINT, sig_int);

    for ( ; ; )
        pause();    /* everything done by children */
}

void sig_int(int signo)
{
    void    pr_cpu_time(void);

    for (int i = 0; i < g_nchildren; i++)
        kill(g_pids[i], SIGTERM);

    while (wait(NULL) > 0)      /* wait for all children */
        ;
    if (errno != ECHILD)
        err_sys("wait error");

    pr_cpu_time();
    exit(0);
}
