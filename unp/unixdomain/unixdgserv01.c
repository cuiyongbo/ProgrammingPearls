#include    "unp.h"

int main(int argc, char **argv)
{
    int sockfd = Socket(AF_LOCAL, SOCK_DGRAM, 0);

    unlink(UNIXDG_PATH);

    struct sockaddr_un servaddr, cliaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_LOCAL;
    strcpy(servaddr.sun_path, UNIXDG_PATH);
    Bind(sockfd, (SA*) &servaddr, sizeof(servaddr));

    dg_echo(sockfd, (SA*) &cliaddr, sizeof(cliaddr));
}
