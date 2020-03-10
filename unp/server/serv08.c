#include "unp_thread.h"
#include "pthread08.h"

static int          g_nthreads;
pthread_mutex_t     g_clifd_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t      g_clifd_cond = PTHREAD_COND_INITIALIZER;

int main(int argc, char** argv)
{
    void thread_make(int);
    void sig_int(int signo);

    int listenfd = 0;
    socklen_t addrlen = 0;
    if(argc == 3)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else if(argc == 4)
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);
    else
        err_quit("Usage: %s [<host>] <#port> <#threads>", argv[0]);

    g_nthreads = atoi(argv[argc-1]);
    g_tptr = (ThreadData*)Calloc(g_nthreads, sizeof(ThreadData));
    g_iget = g_iput = 0;

    for (int i = 0; i < g_nthreads; ++i)
        thread_make(i);

    Signal(SIGINT, sig_int);

    SA* cliaddr = (SA*)Malloc(addrlen);
    while(1)
    {
        socklen_t clilen = addrlen;
        int connfd = Accept(listenfd, cliaddr, &clilen);

        Pthread_mutex_lock(&g_clifd_mutex);
        g_clifd[g_iput] = connfd;
        if(++g_iput == MAXNCLI)
            g_iput = 0;
        if(g_iput == g_iget)
            err_quit("g_iput = g_iget = %d\n", g_iput);
        Pthread_cond_signal(&g_clifd_cond);
        Pthread_mutex_unlock(&g_clifd_mutex);
    }
}

void sig_int(int signo)
{
    void pr_cpu_time(void);
    pr_cpu_time();

    for (int i = 0; i < g_nthreads; ++i)
        printf("thread %d, %ld connections\n", i, g_tptr[i].thread_count);

    exit(0);
}
