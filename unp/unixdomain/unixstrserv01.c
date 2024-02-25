#include    "unp.h"

int main(int argc, char **argv)
{
    void sig_child(int);

    int listenfd = Socket(AF_LOCAL, SOCK_STREAM, 0);

    unlink(UNIXSTR_PATH);

    struct sockaddr_un cliaddr, servaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_LOCAL;
    strcpy(servaddr.sun_path, UNIXSTR_PATH);

    Bind(listenfd, (SA*)&servaddr, sizeof(servaddr));

    Listen(listenfd, LISTEN_QUEUE_LEN);

    Signal(SIGCHLD, sig_child);

    for ( ; ; )
    {
        socklen_t clilen = sizeof(cliaddr);
        int connfd = accept(listenfd, (SA *)&cliaddr, &clilen);
        if (connfd < 0)
        {
            if (errno != EINTR)
                err_sys("accept error");
        }

        // printf("receive connection from %s\n", cliaddr.sun_path);

        pid_t childpid = Fork();
        if(childpid == 0)
        { /* child process */
            Close(listenfd);
            str_echo(connfd);
            exit(0);
        }
        Close(connfd);          /* parent closes connected socket */
    }
}
