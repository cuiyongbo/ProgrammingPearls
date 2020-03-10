#include    "unp.h"

int my_open(const char *, int);

int main(int argc, char **argv)
{
    if(argc != 2)
	{
        err_quit("usage: mycat <pathname>");
	}

    int fd = my_open(argv[1], O_RDONLY);
    if(fd < 0)
	{
        err_sys("cannot open %s", argv[1]);
	}

	printf("receive fd: %d\n", fd);

    ssize_t n;
    char buff[BUFFSIZE];
    while ((n = Read(fd, buff, BUFFSIZE)) > 0)
	{
        Write(STDOUT_FILENO, buff, n);
	}
	
    exit(0);
}
