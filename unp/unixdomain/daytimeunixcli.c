#include    "unp.h"

int main(int argc, char **argv)
{
    int sockfd = Socket(AF_LOCAL, SOCK_STREAM, 0);

    char tmp_path[32] = "/tmp/temp.XXXXXX";
    struct sockaddr_un cliaddr;
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_LOCAL;
    strcpy(cliaddr.sun_path, mktemp(tmp_path));
    Bind(sockfd, (SA*)&cliaddr, sizeof(cliaddr));

    struct sockaddr_un servaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_LOCAL;
    strcpy(servaddr.sun_path, UNIXSTR_PATH);
    Connect(sockfd, (SA *)&servaddr, sizeof(servaddr));

    sleep(5);

    ssize_t n = 0;
    char recvline[MAXLINE + 1];
    while((n = Read(sockfd, recvline, MAXLINE)) > 0)
    {
        printf("receive %d bytes\n", (int)n);
        recvline[n] = 0;    /* null terminate */
        Fputs(recvline, stdout);
    }
    exit(0);
}
