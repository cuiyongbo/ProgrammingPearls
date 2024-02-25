#include    "unp.h"

static int      g_sockfd;

#define QUEUE_LEN      8
#define MAX_DATAGRAM_SIZE   4096

typedef struct
{
  void      *dg_data;       /* ptr to actual datagram */
  size_t    dg_len;         /* length of datagram */
  struct sockaddr  *dg_sa;  /* ptr to sockaddr{} w/client's address */
  socklen_t dg_salen;       /* length of sockaddr{} */
} ClientDatagram;

static ClientDatagram   g_dg[QUEUE_LEN];          /* queue of datagrams to process */
static long g_cntread[QUEUE_LEN+1];   /* diagnostic counter */

static int  g_iget;       /* next one for main loop to process */
static int  g_iput;       /* next one for signal handler to read into */
static int  g_nqueue;     /* # on queue for main loop to process */
static socklen_t g_clilen;/* max length of sockaddr{} */

static void sig_io(int signo)
{
    int nread;
    for (nread = 0; ; )
    {
        if (g_nqueue >= QUEUE_LEN)
            err_quit("receive overflow");

        ClientDatagram* ptr = &g_dg[g_iput];
        ptr->dg_salen = g_clilen;
        ssize_t len = recvfrom(g_sockfd, ptr->dg_data, MAX_DATAGRAM_SIZE,
                                0, ptr->dg_sa, &ptr->dg_salen);
        if (len < 0)
        {
            if (errno == EWOULDBLOCK)
                break;      /* all done; no more queued to read */
            else
                err_sys("recvfrom error");
        }
        ptr->dg_len = len;

        nread++;
        g_nqueue++;
        if (++g_iput >= QUEUE_LEN)
            g_iput = 0;

    }
    g_cntread[nread]++;       /* histogram of # datagrams read per signal */
}

static void sig_hup(int signo)
{
    for (int i = 0; i <= QUEUE_LEN; i++)
        printf("g_cntread[%d] = %ld\n", i, g_cntread[i]);
}

void dg_echo(int sockfd, SA *pcliaddr, socklen_t clilen)
{
    g_sockfd = sockfd;
    g_clilen = clilen;
    g_iget = g_iput = g_nqueue = 0;
    for (int i = 0; i < QUEUE_LEN; i++)
    {
        g_dg[i].dg_data = Malloc(MAX_DATAGRAM_SIZE);
        g_dg[i].dg_sa = Malloc(g_clilen);
        g_dg[i].dg_salen = g_clilen;
    }

    Signal(SIGHUP, sig_hup);
    Signal(SIGIO, sig_io);

    Fcntl(g_sockfd, F_SETOWN, getpid());

    const int on = 1;
    Ioctl(g_sockfd, FIOASYNC, (void*)&on);
    Ioctl(g_sockfd, FIONBIO, (void*)&on);

    sigset_t zeromask, newmask, oldmask;
    Sigemptyset(&zeromask);     /* init three signal sets */
    Sigemptyset(&oldmask);
    Sigemptyset(&newmask);
    Sigaddset(&newmask, SIGIO); /* signal we want to block */

    Sigprocmask(SIG_BLOCK, &newmask, &oldmask);
    for ( ; ; )
    {
        while (g_nqueue == 0)
            sigsuspend(&zeromask);  /* wait for datagram to process */

        Sigprocmask(SIG_SETMASK, &oldmask, NULL);

        Sendto(g_sockfd, g_dg[g_iget].dg_data, g_dg[g_iget].dg_len, 0,
               g_dg[g_iget].dg_sa, g_dg[g_iget].dg_salen);

        if (++g_iget >= QUEUE_LEN)
            g_iget = 0;

        Sigprocmask(SIG_BLOCK, &newmask, &oldmask);
        g_nqueue--;
    }
}
