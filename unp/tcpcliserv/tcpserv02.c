#include "unp.h"

int main(int argc, char** argv)
{
    int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if(listenFd < 0)
    {
        err_sys("socket error");
    }

    struct sockaddr_in servAddr;
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(SERVER_PORT);
    if(bind(listenFd, (SA*)&servAddr, sizeof(servAddr)) < 0)
    {
        err_sys("bind error");
    }

    if(listen(listenFd, LISTEN_QUEUE_LEN) < 0)
    {
        err_sys("listen error");
    }

    if(Signal(SIGCHLD, sig_child) == SIG_ERR)
    {
        err_sys("Signal error");
    }

    socklen_t cliLen;
    struct sockaddr_in cliAddr;
    while(1)
    {
        cliLen = sizeof(cliAddr);
        int cliFd = accept(listenFd, (SA*)&cliAddr, &cliLen);
        if(cliFd < 0)
        {
            err_msg("accept error");
            continue;
        }

        pid_t pid = fork();
        if(pid < 0)
        {
            err_msg("fork error");
            close(cliFd);
            continue;
        }
        else if(pid == 0)
        {
            close(listenFd);
            str_echo(cliFd);
            exit(EXIT_SUCCESS);
        }
        close(cliFd);
    }
}
