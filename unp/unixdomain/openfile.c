#include    "unp.h"

int main(int argc, char **argv)
{
    if(argc != 4)
	{
        err_quit("%s <sockfd#> <filename> <mode>", argv[0]);
	}

    int fd = open(argv[2], atoi(argv[3]));
    if(fd < 0)
	{
        exit((errno > 0) ? errno : 255 );
	}

	printf("send fd: %d\n", fd);

    if(Write_fd(atoi(argv[1]), "", 1, fd) < 0)
	{
        exit((errno > 0) ? errno : 255 );
	}

    exit(0);
}
