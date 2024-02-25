#include "apue.h"

int main()
{
	struct tm* tmp;
	struct timespec tout;
	int err;
	char buf[BUFSIZ];
	pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
	pthread_mutex_lock(&lock);
	printf("mutex locked\n");
	clock_gettime(CLOCK_REALTIME, &tout);
	tmp = localtime(&tout.tv_sec);
	strftime(buf, sizeof(buf), "%r", tmp);
	printf("current time: %s\n", buf);
	tout.tv_sec += 10;
	err = pthread_mutex_timedlock(&lock, &tout);	
	clock_gettime(CLOCK_REALTIME, &tout);
	tmp = localtime(&tout.tv_sec);
	strftime(buf, sizeof(buf), "%r", tmp);
	printf("current time: %s\n", buf);
	if(err == 0)
		printf("mutex locked again\n");
	else
		printf("can't lock mutex again: %s\n", strerror(err));
	return 0;
}
