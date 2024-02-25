#include "apue.h"

int main()
{
	char buf1[64];
	char buf2[64];
	time_t t = time(NULL);
	struct tm* tmp = localtime(&t);
	if(strftime(buf1, 16, "time and date: %r, %a %b %d, %Y", tmp) == 0)
		printf("buffer length 16 is too small\n");
	else
		printf("%s\n", buf1);

	if(strftime(buf2, 64, "time and date: %r, %a %b %d, %Y", tmp) == 0)
		printf("buffer length 64 is too small\n");
	else
		printf("%s\n", buf2);

	return 0;
}
