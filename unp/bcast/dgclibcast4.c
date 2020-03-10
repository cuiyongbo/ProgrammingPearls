#include    "unp.h"

static void recvfrom_alarm(int)
{
    return;     /* just interrupt the recvfrom() */
}

void dg_cli(FILE *fp, int sockfd, const SA *pservaddr, socklen_t servlen)
{
    char sendline[MAXLINE], recvline[MAXLINE + 1];
    struct sockaddr* preply_addr = (struct sockaddr*)Malloc(servlen);

    const int on = 1;
    Setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));

    fd_set rset;
    sigset_t sigset_alrm, emptyset;
    Sigemptyset(&emptyset);
    Sigemptyset(&sigset_alrm);
    Sigaddset(&sigset_alrm, SIGALRM);

    Signal(SIGALRM, recvfrom_alarm);

    while (Fgets(sendline, MAXLINE, fp) != NULL) {

        Sendto(sockfd, sendline, strlen(sendline), 0, pservaddr, servlen);

        Sigprocmask(SIG_BLOCK, &sigset_alrm, NULL);
        alarm(5);

        for ( ; ; ) {
            FD_CLR(&rset);
            FD_SET(sockfd, &rset);
            int n = pselect(sockfd+1, &rset, NULL, NULL, NULL, &emptyset);
            if (n < 0)
            {
                if (errno == EINTR)
                    break;
                else
                    err_sys("pselect error");
            } else if (n != 1)
                err_sys("pselect error: returned %d", n);

            socklen_t len = servlen;
            n = recvfrom(sockfd, recvline, MAXLINE, 0, preply_addr, &len);
            recvline[n] = 0;    /* null terminate */
            printf("from %s: %s", Sock_ntop_host(preply_addr, len), recvline);
        }
    }
    free(preply_addr);
}
