#include    "unp.h"

#define MAXN    16384       /* max # bytes client can request */

void web_child(int sockfd)
{
    ssize_t  n = 0;
    char line[MAXLINE], result[MAXN];
    for(;;)
    {
        //if (Readline(sockfd, line, MAXLINE) == 0)
        n = Read(sockfd, line, MAXLINE);
        if (n == 0)
            return;     /* connection closed by other end */

        line[n] = 0;
        // printf("client asks for %s", line);
        int ntowrite = atol(line);
        if ((ntowrite <= 0) || (ntowrite > MAXN))
            err_quit("client request for %d bytes", ntowrite);

        Writen(sockfd, result, ntowrite);
    }
}
