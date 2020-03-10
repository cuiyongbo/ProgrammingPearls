#include    "unp.h"

int main(int argc, char **argv)
{
    int sockfd = Socket(AF_LOCAL, SOCK_STREAM, 0);

    struct sockaddr_un servaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_LOCAL;
    strcpy(servaddr.sun_path, UNIXSTR_PATH);

    Connect(sockfd, (SA*)&servaddr, sizeof(servaddr));

    str_cli(stdin, sockfd);

    exit(0);
}
