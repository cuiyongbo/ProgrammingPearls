#include "apue.h"
#include <poll.h>
#include <pthread.h>
#include <sys/msg.h>
#include <sys/socket.h>

#define QUEUE_COUNT	3
#define MAX_MSG_SIZE	512
#define KEY 0x123  /* key for first message queue */

typedef struct QueueMsg
{
	long type;
	char msg[MAX_MSG_SIZE];
} QueueMsg;

int main(int argc, char* argv[])
{
	if(argc != 3)
		err_quit("Usage: %s KEY message", argv[0]);

	key_t key = strtol(argv[1], NULL, 0);
	long queueId = msgget(key, 0666);
	if(queueId < 0)
		err_sys("can't open queue<%ld>", (long)key);

	QueueMsg qmsg;
	memset(&qmsg, 0, sizeof(qmsg));
	strncpy(qmsg.msg, argv[2], MAX_MSG_SIZE - 1);
	qmsg.type = 1;
	if(msgsnd(queueId, &qmsg, strlen(qmsg.msg), 0) < 0)
		err_sys("msgsnd error");

	return 0;	
}

