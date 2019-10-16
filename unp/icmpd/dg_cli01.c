#include "unp_icmpd.h"

void dg_cli(FILE *fp, int sockfd, const SA *pservaddr, socklen_t servlen)
{
    Sock_bind_wild(sockfd, pservaddr->sa_family);

    int icmpfd = Socket(AF_LOCAL, SOCK_STREAM, 0);
    struct sockaddr_un sun;
    bzero(&sun, sizeof(sun));
    sun.sun_family = AF_LOCAL;
    strcpy(sun.sun_path, ICMPD_PATH);
    Connect(icmpfd, (SA*)&sun, sizeof(sun));
    Write_fd(icmpfd, "1", 1, sockfd);

    char sendline[MAXLINE], recvline[MAXLINE + 1];
    ssize_t n = Read(icmpfd, recvline, 1);
    if(n != 1 || recvline[0] != '1')
    {
        err_quite("error creating icmp socket, n=%d, char=%c",
            n, recvline[0]);
    }

    fd_set rset;
    FD_ZERO(&rset);
    struct timeval tv;
    struct Icmpd_error icmpd_err;
    int maxfdp1 = max(sockfd, icmpfd) + 1;
    while(Fgets(sendline, MAXLINE, fp) != NULL)
    {
        Sendto(sockfd, sendline, strlen(sendline), 0, pservaddr, servlen);

        tv.tv_sec = 5;
        tv.tv_usec = 0;
        FD_SET(sockfd, &rset);
        FD_SET(icmpfd, &rset);
        if (Select(maxfdp1, &rset, NULL, NULL, &tv) == 0)
        {
            fprintf(stderr, "socket timeout\n");
            continue;
        }

        if (FD_ISSET(sockfd, &rset))
        {
            n = Recvfrom(sockfd, recvline, MAXLINE, 0, NULL, NULL);
            recvline[n] = 0;    /* null terminate */
            Fputs(recvline, stdout);
        }

        if (FD_ISSET(icmpfd, &rset))
        {
            n = Read(icmpfd, &icmpd_err, sizeof(icmpd_err));
            if(n == 0)
                err_quit("ICMP daemon terminated");
            else if(n != sizeof(icmpd_err))
                err_quit("n = %d, expected %d", n, sizeof(icmpd_err));

            printf("ICMP error: dest = %s, %s, type = %d, code = %d\n",
                   Sock_ntop_host(&icmpd_err.icmpd_dest, icmpd_err.icmpd_len),
                   strerror(icmpd_err.icmpd_errno),
                   icmpd_err.icmpd_type, icmpd_err.icmpd_code);
        }
    }
}
