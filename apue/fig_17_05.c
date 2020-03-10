#include "apue.h"
#include <sys/un.h>
#include <sys/socket.h>

int main()
{
	const char* path = "/tmp/foo.socket";
	unlink(path);
	struct sockaddr_un un;
	strcpy(un.sun_path, path);
	int fd = socket(AF_LOCAL, SOCK_STREAM, 0);
	if(fd < 0)
		err_sys("socket error");
	un.sun_family = AF_LOCAL;
	socklen_t size = offsetof(struct sockaddr_un, sun_path) + strlen(un.sun_path);
	if(bind(fd, (struct sockaddr*)&un, size) < 0)
		err_sys("bind error");
	printf("UNIX domain socket bound\n");
	return 0;
}

