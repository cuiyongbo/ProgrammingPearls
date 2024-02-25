#include "apue.h"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        err_quit("Usage: %s <string>\n", argv[0]);
    } 
    
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        err_sys("pipe failed");
    }

    pid_t cpid = fork();
    if (cpid == -1) {
        err_sys("fork failed");
    }

    if (cpid == 0) {    /* Child reads from pipe */
        close(pipefd[1]);          /* Close unused write end */

        char buf;
        while (read(pipefd[0], &buf, 1) > 0)
            write(STDOUT_FILENO, &buf, 1);

        write(STDOUT_FILENO, "\n", 1);
        close(pipefd[0]);
        _exit(EXIT_SUCCESS);           
    } else {                      /* Parent writes argv[1] to pipe */
        close(pipefd[0]);          /* Close unused read end */
        write(pipefd[1], argv[1], strlen(argv[1]));
        close(pipefd[1]);          /* Reader will see EOF */
        wait(NULL);                /* Wait for child */
        exit(EXIT_SUCCESS);
    }
}