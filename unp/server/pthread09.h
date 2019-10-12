#pragma once

typedef struct
{
    pthread_t thread_id;
    long thread_count;
} ThreadData;

ThreadData* g_tptr;

int g_listenfd, g_nthreads;
socklen_t g_addrlen;
pthread_mutex_t g_mlock;
