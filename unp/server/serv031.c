#include    "unp.h"

static int      g_nchildren;
static pid_t*   g_pids;
long* g_cptr;
long* meter(int);

int main(int argc, char **argv)
{
    void        sig_int(int);
    pid_t       child_make(int, int, int);
    void my_lock_init(char *pathname);

    int         listenfd = 0;
    socklen_t   addrlen = 0;
    if (argc == 3)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else if (argc == 4)
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);
    else
        err_quit("usage: %s [ <host> ] <port#> <#children>", argv[0]);

    g_nchildren = atoi(argv[argc-1]);
    g_pids = (pid_t*)Calloc(g_nchildren, sizeof(pid_t));
    g_cptr = meter(g_nchildren);

    my_lock_init("/tmp/lock.XXXXXX");

    for (int i = 0; i < g_nchildren; i++)
        g_pids[i] = child_make(i, listenfd, addrlen); /* parent returns */

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

    for(int i=0; i<g_nchildren; ++i)
        printf("child %d, %ld connections\n", i, g_cptr[i]);

    exit(0);
}
