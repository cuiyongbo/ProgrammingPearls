#include "unp_thread.h"
#include "pthread07.h"

void sig_int(int signo)
{
    void pr_cpu_time(void);
    pr_cpu_time();

    for (int i = 0; i < g_nthreads; ++i)
        printf("thread %d, %ld connections\n", i, g_tptr[i].thread_count);

    exit(0);
}

pthread_mutex_t g_mlock = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char** argv)
{
    void thread_make(int);

    if(argc == 3)
        g_listenfd = Tcp_listen(NULL, argv[1], &g_addrlen);
    else if(argc == 4)
        g_listenfd = Tcp_listen(argv[1], argv[2], &g_addrlen);
    else
        err_quit("Usage: %s [<host>] <#port> <#threads>", argv[0]);

    g_nthreads = atoi(argv[argc-1]);
    g_tptr = (ThreadData*)Calloc(g_nthreads, sizeof(ThreadData));

    for (int i = 0; i < g_nthreads; ++i)
        thread_make(i);

    Signal(SIGINT, sig_int);

    for(;;)
        pause();
}

