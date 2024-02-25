#pragma once

typedef struct
{
    pthread_t thread_id;
    long thread_count;
} ThreadData;

ThreadData* g_tptr;

#define MAXNCLI 32
int                 g_clifd[MAXNCLI], g_iget, g_iput;
pthread_mutex_t     g_clifd_mutex;
pthread_cond_t      g_clifd_cond;
