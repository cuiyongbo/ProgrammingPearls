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
     /* Don't rely on the value of tv now! */

     if (retval == -1)
         perror("select()");
     else if (retval) {
         printf("Data is available now.\n");
         int isReady = FD_ISSET(0, &rfds);
         printf("stdin is %s\n", isReady ? "ready" : "not ready");
     }
     else
         printf("No data within five seconds.\n");

     exit(EXIT_SUCCESS);
 }

