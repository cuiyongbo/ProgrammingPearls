#include "apue.h"

int g_counter = 0;
int g_counterLimit = 100;
pthread_mutex_t g_counterMutex;
pthread_cond_t g_counterCond;

void* threadFunc(void* data)
{
    while(1)
    {    
        pthread_mutex_lock(&g_counterMutex);
        ++g_counter;
        if(g_counter == g_counterLimit)
        {
            pthread_cond_signal(&g_counterCond);
            printf("%#x: limit reached, send the signal\n", (int)pthread_self());
            pthread_mutex_unlock(&g_counterMutex);
            pthread_exit(NULL);
        }
        else
        {
            pthread_mutex_unlock(&g_counterMutex);
        }
        sleep(1);
    }
} 

int main()
{
    pthread_mutex_init(&g_counterMutex, NULL);
    pthread_cond_init(&g_counterCond, NULL);

    pthread_t tid;
    int err = pthread_create(&tid, NULL, threadFunc, NULL);
    if(err != 0)
    {
        printf("pthread_create error: %s", strerror(err));
    }

    pthread_mutex_lock(&g_counterMutex);
    while(g_counter != g_counterLimit)
    {
        err = pthread_cond_wait(&g_counterCond, &g_counterMutex);
        if(err != 0)
        {
            printf("pthread_cond_wait error: %s", strerror(err));
            pthread_mutex_unlock(&g_counterMutex);
            exit(EXIT_FAILURE);
        }
        else
        {
            printf("signal received\n");
        }
    }
    pthread_mutex_unlock(&g_counterMutex);

    err = pthread_join(tid, NULL);
    if(err != 0)
    {
        printf("pthread_join error: %s", strerror(err));
    }

    pthread_mutex_destroy(&g_counterMutex);
    pthread_cond_destroy(&g_counterCond);
}
