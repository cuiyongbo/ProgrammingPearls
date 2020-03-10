#include    "unp_thread.h"
#include    "pthread08.h"

void thread_make(int i)
{
    void* thread_main(void*);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    Pthread_create(&g_tptr[i].thread_id, &attr, thread_main, (void*)i);
    pthread_attr_destroy(&attr);
    return;
}

void* thread_main(void *arg)
{
    void    web_child(int);

    int thread_idx = (int)arg;
    printf("thread %d starting\n", thread_idx);
    for (;;)
    {
        Pthread_mutex_lock(&g_clifd_mutex);

        while (g_iget == g_iput)
            Pthread_cond_wait(&g_clifd_cond, &g_clifd_mutex);

        int connfd = g_clifd[g_iget];
        if (++g_iget == MAXNCLI)
            g_iget = 0;

        Pthread_mutex_unlock(&g_clifd_mutex);

        g_tptr[thread_idx].thread_count++;

        web_child(connfd);
        Close(connfd);
    }
}
