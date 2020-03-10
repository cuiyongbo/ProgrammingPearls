#include    "unp.h"

int main(int argc, char **argv)
{
    int sockfd = Socket(AF_LOCAL, SOCK_DGRAM, 0);

    char tmp_path[32] = "/tmp/temp.XXXXXX";
    struct sockaddr_un cliaddr;
    bzero(&cliaddr, sizeof(cliaddr));       /* bind an address for us */
    cliaddr.sun_family = AF_LOCAL;
    strcpy(cliaddr.sun_path, mktemp(tmp_path));

    // sending a datagram on an unbound Unix domain datagram socket does not
    // implicitly bind a pathname to the socket. Therefore, if we omit this step,
    // the server's call to recvfrom in the dg_echo function returns a null pathname,
    // which then causes an error when the server calls sendto.
    // refer to section 15.4 for futher details
    Bind(sockfd, (SA*)&cliaddr, sizeof(cliaddr));

    struct sockaddr_un servaddr;
    bzero(&servaddr, sizeof(servaddr)); /* fill in server's address */
    servaddr.sun_family = AF_LOCAL;
    strcpy(servaddr.sun_path, UNIXDG_PATH);

    dg_cli(stdin, sockfd, (SA*)&servaddr, sizeof(servaddr));

    exit(0);
}
