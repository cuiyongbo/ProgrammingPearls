#include "unp.h"

int main(int argc, char** argv)
{
    if(argc != 3)
        err_quit("Usage: %s <host> <port#>\n", argv[0]);

    int sockFd = Tcp_connect(argv[1], argv[2]);

    int size;
    socklen_t len;
    Getsockopt(sockFd, SOL_SOCKET, SO_SNDBUF, &size, &len);
    printf("Current limit: %d\n", size);

    int sendBufferSize = 32768;
    Setsockopt(sockFd, SOL_SOCKET, SO_SNDBUF, &sendBufferSize, sizeof(sendBufferSize));

    Getsockopt(sockFd, SOL_SOCKET, SO_SNDBUF, &size, &len);
    printf("Current limit: %d\n", size);

    size = sendBufferSize/2;
    void* buff = Malloc(size);
    Write(sockFd, buff, size);
    printf("wrote %d bytes of normal data\n", size);
    Send(sockFd, "a", 1, MSG_OOB);
    printf("wrote 1 byte of OOB data\n");
    Write(sockFd, buff, 1024);
    printf("wrote 1024 bytes of normal data\n");
    free(buff);

    return 0;
}
