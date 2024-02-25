#include "unp.h"

int main(int argc, char** argv)
{
    if(argc != 3)
        err_quit("Usage: %s <host> <port#>\n", argv[0]);

    int sockFd = Tcp_connect(argv[1], argv[2]);

    Write(sockFd, "123", 3);
    printf("wrote 3 bytes of normal data\n");
    sleep(1);

    Send(sockFd, "4", 1, MSG_OOB);
    printf("wrote 1 byte of OOB data\n");
    sleep(1);

    Write(sockFd, "56", 2);
    printf("wrote 2 bytes of normal data\n");
    sleep(1);

    Send(sockFd, "7", 1, MSG_OOB);
    printf("wrote 1 byte of OOB data\n");
    sleep(1);

    Write(sockFd, "89", 2);
    printf("wrote 2 bytes of normal data\n");
    sleep(1);

    return 0;
}
