#include "apue.h"

void test(const struct timeval* t)
{
	t->tv_sec++;
}

int main()
{
	struct timeval t {50, 500};
	test(&t);
	printf("timeval tv_sec: %d, tv_usec: %d\n", (int)t.tv_sec, (int)t.tv_usec);
	printf("FD_SETSIZE: %d\n", FD_SETSIZE);
	t.tv_sec = 10;
	t.tv_usec = 200;
	select(0, NULL, NULL, NULL, &t);
	printf("timeval tv_sec: %d, tv_usec: %d\n", (int)t.tv_sec, (int)t.tv_usec);
}
