#include "apue.h"
#include <poll.h>
#include <pthread.h>
#include <sys/msg.h>
#include <sys/socket.h>

#define QUEUE_COUNT	3
#define MAX_MSG_SIZE	512
#define KEY 0x123  /* key for first message queue */

typedef struct Threadinfo
{
	int qid;
	int fd;	
} Threadinfo;

typedef struct QueueMsg
{
	long type;
	char msg[MAX_MSG_SIZE];
} QueueMsg;

void* threadFunc(void* arg)
{
	QueueMsg qmsg;
	Threadinfo* tip = arg;	
	for(;;)
	{
		memset(&qmsg, 0, sizeof(qmsg));
		int n = msgrcv(tip->qid, &qmsg, MAX_MSG_SIZE, 0, MSG_NOERROR);
		if(n < 0)
			err_sys("msgrcv error");
		if(write(tip->fd, qmsg.msg, n) < 0)
			err_sys("write error");
	}
	pthread_exit(NULL);
}

int main()
{
	int fd[2];
	int qidArray[QUEUE_COUNT];
	Threadinfo tinfoArray[QUEUE_COUNT];
	struct pollfd pfd[QUEUE_COUNT];
	pthread_t tidArray[QUEUE_COUNT];
	for(int i=0; i<QUEUE_COUNT; i++)
	{
		qidArray[i] = msgget(KEY+i, IPC_CREAT|0666);
		if(qidArray[i] < 0)
			err_sys("msgget error");
		printf("queue %d ID %d\n", i, qidArray[i]);
		
		if(socketpair(AF_LOCAL, SOCK_DGRAM, 0, fd) < 0)
			err_sys("socketpair error");
		
		pfd[i].fd = fd[0];
		pfd[i].events = POLLIN;
		tinfoArray[i].qid = qidArray[i];
		tinfoArray[i].fd = fd[1];
		int err = pthread_create(&tidArray[i], NULL, threadFunc, &tinfoArray[i]);
		if(err < 0)
			err_exit(err, "pthread_create error");
	}

	char buf[MAX_MSG_SIZE];
	for(;;)
	{
		if(poll(pfd, QUEUE_COUNT, -1) < 0)
			err_sys("poll error");

		for(int i=0; i<QUEUE_COUNT; i++)
		{
			if(pfd[i].revents & POLLIN)
			{
				int n = read(pfd[i].fd, buf, MAX_MSG_SIZE);
				if(n < 0)
					err_sys("read error");
				buf[n] = 0;
				printf("queue id<%d>, message: %s\n", qidArray[i], buf);
			}
		}
	}

	for(int i=0; i<QUEUE_COUNT; i++)
	{
		if(msgctl(qidArray[i], IPC_RMID, NULL) != 0)
			err_msg("failed to remove message queue<%d>", qidArray[i]);		
	}

	return 0;	
}

