#include "opend.h"

#define NALLOC 10

int clientCount;
ClientInfo* clientInfos;

static void client_alloc()
{
	if(clientInfos == NULL)
		clientInfos = malloc(NALLOC * sizeof(ClientInfo));
	else
		clientInfos = realloc(clientInfos, (clientCount + NALLOC) * sizeof(ClientInfo));

	if(clientInfos == NULL)
		err_sys("Failed to allocate client information array");

	for(int i=clientCount; i<clientCount+NALLOC; i++)
		clientInfos[i].fd = -1;

	clientCount += NALLOC;
}

int client_add(int fd, uid_t uid)
{
	if(clientInfos == NULL)
		client_alloc();

again:
	for(int i=0; i<clientCount; i++)
	{
		if(clientInfos[i].fd == -1)
		{
			clientInfos[i].fd = fd;
			clientInfos[i].uid = uid;
			return i;
		}
	}

	client_alloc();
	goto again;
}

void client_del(int fd)
{
	for(int i=0; i<clientCount; i++)
	{
		if(clientInfos[i].fd == fd)
		{
			clientInfos[i].fd = -1;
			return;
		}
	}
	err_quit("can't find client entry for fd: %d", fd);
}

