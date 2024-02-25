#include "unp.h"

int main(int argc, char** argv)
{
    if(argc != 3)
        err_quit("Usage: %s <host> <port#>\n", argv[0]);

    int sockFd = Tcp_connect(argv[1], argv[2]);

    Write(sockFd, "123", 3);
    printf("wrote 3 bytes of normal data\n");
    Send(sockFd, "4", 1, MSG_OOB);
    printf("wrote 1 byte of OOB data\n");
    Write(sockFd, "5", 1);
    printf("wrote 1 byte of normal data\n");
    Send(sockFd, "6", 1, MSG_OOB);
    printf("wrote 1 byte of OOB data\n");
    Write(sockFd, "7", 1);
    printf("wrote 1 byte of normal data\n");
    sleep(1);

    return 0;
}
