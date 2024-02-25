#include    "unp.h"

#define MAXN    16384       /* max # bytes to request from server */

int main(int argc, char **argv)
{
    if (argc != 6)
        err_quit("Usage: %s <hostname or IPaddr> <port> <#children> "
                 "<#loops/child> <#bytes/request>", argv[0]);

    int nchildren = atoi(argv[3]);
    int nloops = atoi(argv[4]);
    int nbytes = atoi(argv[5]);
    char request[MAXLINE], reply[MAXN];
    snprintf(request, sizeof(request), "%d\n", nbytes); /* newline at end */
    size_t len = strlen(request);

    for (int i = 0; i < nchildren; i++)
    {
        pid_t pid = Fork();
        if (pid == 0)
        {
            for (int j = 0; j < nloops; j++)
            {
                int fd = Tcp_connect(argv[1], argv[2]);

                Write(fd, request, len);

                ssize_t n = Readn(fd, reply, nbytes);
                if (n != nbytes)
                    err_quit("server returned %d bytes", n);

                Close(fd);      /* TIME_WAIT on client, not server */
            }
            printf("child %d done\n", i);
            exit(0);
        }
    }

    while (wait(NULL) > 0)  /* now parent waits for all children */
        ;

    if (errno != ECHILD)
        err_sys("wait error");

    exit(0);
}
