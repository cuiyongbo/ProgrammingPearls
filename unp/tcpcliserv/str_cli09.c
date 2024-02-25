#include "unp.h"
#include "sum.h"

void str_cli(FILE* fp, int sockfd)
{
    char sendBuf[MAXLINE];
    struct Argument args;
    struct Result result;
    while(fgets(sendBuf, MAXLINE, fp) != NULL)
    {
        if (sscanf(sendBuf, "%ld %ld", &args.arg1, &args.arg2) != 2)
        {
            printf("invalid input\n");
            continue;
        }

        Writen(sockfd, &args, sizeof(args));

        if(read(sockfd, &result, sizeof(result)) == 0)
            err_quit("str_cli: server terminated prematurely");

        printf("%ld\n", result.sum);
    }
}
