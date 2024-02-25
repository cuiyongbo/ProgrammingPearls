#include "unp.h"

void handler(int signo)
{
    printf("process <%d>: %s <%d> signal received\n", (int)getpid(), strsignal(signo), signo);
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        err_quit("Usage: %s server_ip", argv[0]);
    }

    Signal(SIGPIPE, handler);

    int sockFd = Socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in servAddr;
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(13);
    if(inet_pton(AF_INET, argv[1], &servAddr.sin_addr) <= 0)
    {
        err_sys("inet_pton error for %s", argv[1]);
    }

    Connect(sockFd, (SA*)&servAddr, sizeof(servAddr));

    sleep(2);
    Write(sockFd, "hello", strlen("hello"));
    sleep(2);
    Write(sockFd, "hello", strlen("hello"));
    sleep(2);
    Write(sockFd, "hello", strlen("hello"));

    return 0;
}
