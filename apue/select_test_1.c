#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

 int main(void)
 {
     /* Watch stdin (fd 0) to see when it has input. */
     fd_set rfds;
     FD_ZERO(&rfds);
     FD_SET(0, &rfds);

     /* Wait up to five seconds. */
     struct timeval tv;
     tv.tv_sec = 5;
     tv.tv_usec = 0;

     int retval = select(1, &rfds, NULL, NULL, &tv);
     if (retval == -1)
         perror("select()");
     else if (retval) {
         printf("Data is available now.\n");
         int isReady = FD_ISSET(0, &rfds);
         printf("stdin is %s\n", isReady ? "ready" : "not ready");
		int bytesRead;
		char buf[256];
		while((bytesRead=read(0, buf, sizeof(buf))) >0)
		{
			buf[bytesRead] = 0;
			printf("%s", buf);
		}
     }
     else
         printf("No data within five seconds.\n");

     exit(EXIT_SUCCESS);
 }

