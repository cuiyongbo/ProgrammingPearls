#include "unp.h"

void insufficient_sig_child_handler(int signo)
{
    pid_t pid = wait(NULL);
    printf("child <%ld> terminated\n", (long)pid);
    return;
}

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

    if(Signal(SIGCHLD, insufficient_sig_child_handler) == SIG_ERR)
    {
        err_sys("signal error");
    }

    socklen_t cliLen;
    struct sockaddr_in cliAddr;
    while(1)
    {
        cliLen = sizeof(cliAddr);
        int cliFd = accept(listenFd, (SA*)&cliAddr, &cliLen);
        if(cliFd < 0)
        {
            if(errno == EINTR)
                continue;
            else
                err_sys("accept error");
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
