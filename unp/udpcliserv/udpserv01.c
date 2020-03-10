#include    "unp.h"

int main(int argc, char** argv)
{
    int sockfd = Socket(AF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in servaddr, cliaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(SERVER_PORT);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    Bind(sockfd, (SA*)&servaddr, sizeof(servaddr));

    dg_echo(sockfd, (SA*)&cliaddr, sizeof(cliaddr));
}
