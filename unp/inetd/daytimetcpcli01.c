#include "unp.h"

int main(int argc, char const *argv[])
{
    if(argc != 3)
        err_quit("usage: %s <hostname/IPaddress> <service/port#>", argv[0]);

    int sockFd = Tcp_connect(argv[1], argv[2]);

    int n = 0;
    char buff[MAXLINE];
    while((n=Read(sockFd, buff, element_of(buff))) > 0)
    {
        buff[n] = 0;
        printf("%s", buff);
    }
    Close(sockFd);
    return 0;
}
