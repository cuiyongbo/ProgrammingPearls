#include "unp_thread.h"
#include "pthread07.h"

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

void* thread_main(void* arg)
{
    void web_child(int);
    printf("thread %d starting\n", (int)arg);
    SA* cliaddr = (SA*)Malloc(g_addrlen);
    while(1)
    {
        Pthread_mutex_lock(&g_mlock);
        socklen_t clilen = g_addrlen;
        int connfd = Accept(g_listenfd, cliaddr, &clilen);
        Pthread_mutex_unlock(&g_mlock);

        g_tptr[(int)arg].thread_count++;

        web_child(connfd);
        Close(connfd);
    }
}
