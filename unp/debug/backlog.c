#include    "unp.h"

#define PORT        9999
#define ADDR        "127.0.0.1"
#define MAXBACKLOG  100

struct sockaddr_in  g_serv;

int g_pipefd[2];
#define g_pfd g_pipefd[1]   /* parent's end */
#define g_cfd g_pipefd[0]   /* child's end */

void    do_parent(void);
void    do_child(void);

int main(int argc, char **argv)
{
    Socketpair(AF_UNIX, SOCK_STREAM, 0, g_pipefd);

    bzero(&g_serv, sizeof(g_serv));
    g_serv.sin_family = AF_INET;
    g_serv.sin_port = htons(PORT);
    Inet_pton(AF_INET, ADDR, &g_serv.sin_addr);

    pid_t pid = Fork();
    if (pid == 0)
        do_child();
    else
        do_parent();

    exit(0);
}

void parent_alrm(int signo)
{
    return;     /* just interrupt blocked connect() */
}

void do_parent(void)
{
    int     backlog, j, junk, fd[MAXBACKLOG + 1];

    Close(g_cfd);
    Signal(SIGALRM, parent_alrm);

    for (backlog = 0; backlog <= 14; backlog++)
    {
        printf("parent backlog = %d: ", backlog);
        Write(g_pfd, &backlog, sizeof(int));  /* tell child value */
        Read(g_pfd, &junk, sizeof(int));      /* wait for child */

        for (j = 1; j <= MAXBACKLOG; j++)
        {
            fd[j] = Socket(AF_INET, SOCK_STREAM, 0);
            alarm(2);
            if (connect(fd[j], (SA*)&g_serv, sizeof(g_serv)) < 0)
            {
                if (errno != EINTR)
                {
                    err_sys("connect error, j = %d", j);
                }

                printf("timeout, %d connections completed\n", j-1);

                for (int k = 1; k <= j; k++)
                {
                    Close(fd[k]);
                }
                break;  /* next value of backlog */
            }
            alarm(0);
        }
        if (j > MAXBACKLOG)
            printf("%d connections?\n", MAXBACKLOG);
    }
    backlog = -1;       /* tell child we're all done */
    Write(g_pfd, &backlog, sizeof(int));
}

void do_child(void)
{
    int junk = -1;
    const int on = 1;

    Close(g_pfd);

    int backlog;
    Read(g_cfd, &backlog, sizeof(int));   /* wait for parent */
    while(backlog >= 0)
    {
        int listenfd = Socket(AF_INET, SOCK_STREAM, 0);
        Setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        Bind(listenfd, (SA*)&g_serv, sizeof(g_serv));
        Listen(listenfd, backlog);      /* start the listen */

        Write(g_cfd, &junk, sizeof(int)); /* tell parent */
        Read(g_cfd, &backlog, sizeof(int));/* just wait for parent */
        Close(listenfd);    /* closes all queued connections, too */
    }
}
