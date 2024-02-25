#pragma once

#include    "unp_icmpd.h"

struct Client
{
  int   connfd;         /* Unix domain stream socket to client */
  int   family;         /* AF_INET or AF_INET6 */
  int   lport;          /* local port bound to client's UDP socket */
                        /* network byte ordered */
} g_client[FD_SETSIZE];

                    /* 4globals */
int             g_fd4, g_fd6, g_listenfd, g_maxi, g_maxfd, g_nready;
fd_set          g_rset, g_allset;
struct sockaddr_un  g_cliaddr;

            /* 4function prototypes */
int      readable_conn(int);
int      readable_listen(void);
int      readable_v4(void);
int      readable_v6(void);
