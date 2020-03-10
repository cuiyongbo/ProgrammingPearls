#include "unp_thread.h"

void* do_it(void*);
void sig_int(int signo);

int main(int argc, char** argv)
{
    int listenfd = 0;
    socklen_t addrlen = 0;
    if(argc == 2)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else if(argc == 3)
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);
    else
        err_quit("Usage: %s [<host>] <#port>", argv[0]);

    Signal(SIGINT, sig_int);

    pthread_t tid;
    SA* cliaddr = (SA*)Malloc(addrlen);
    while(1)
    {
        socklen_t clilen = addrlen;
        int connfd = Accept(listenfd, cliaddr, &clilen);
        Pthread_create(&tid, NULL, &do_it, (void*)connfd);
        printf("create child %#lx\n", (long)tid);
    }
}

void* do_it(void* arg)
{
    void web_child(int);

    Pthread_detach(pthread_self());

    int connfd = (int)arg;
    web_child(connfd);
    Close(connfd);
    return NULL;
}

void sig_int(int signo)
{
    void pr_cpu_time(void);
    pr_cpu_time();
    exit(0);
}
