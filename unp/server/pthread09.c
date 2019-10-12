#include "unp_thread.h"
#include "pthread09.h"

void thread_make(int i)
{
    void* thread_main(void*);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    Pthread_create(&g_tptr[i].thread_id, &attr, thread_main, (void*)i);
    pthread_attr_destroy(&attr);
}

void* thread_main(void* arg)
{
    void web_child(int);
    printf("thread %d starting\n", (int)arg);
    SA* cliaddr = (SA*)Malloc(g_addrlen);
    while(1)
    {
        socklen_t clilen = g_addrlen;
        int connfd = Accept(g_listenfd, cliaddr, &clilen);

        g_tptr[(int)arg].thread_count++;

        web_child(connfd);
        Close(connfd);
    }
}
