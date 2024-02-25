#include    "web.h"

void write_get_cmd(struct File* fptr)
{
    char line[MAXLINE];
    int n = snprintf(line, sizeof(line), GET_CMD, fptr->f_name);
    Writen(fptr->f_fd, line, n);
    printf("wrote %d bytes for %s\n", n, fptr->f_name);

    fptr->f_flags = F_READING;
    FD_SET(fptr->f_fd, &g_rset);
    g_maxfd = max(g_maxfd, fptr->f_fd);
}
