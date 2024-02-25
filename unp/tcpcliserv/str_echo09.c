#include "unp.h"
#include "sum.h"

void str_echo(int sockFd)
{
    struct Argument args;
    struct Result result;
    for(;;)
    {
        if(read(sockFd, &args, sizeof(args)) == 0)
            return;

        result.sum = args.arg1 + args.arg2;
        Writen(sockFd, &result, sizeof(result));
    }
}
