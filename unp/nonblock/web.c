#include    "web.h"

int main(int argc, char **argv)
{
    int     i;
    char    buf[MAXLINE];

    if (argc < 5)
        err_quit("usage: %s <#conns> <hostname> <homepage> <file1> ...", argv[0]);

    int max_nconn = atoi(argv[1]);

    g_nfiles = min(argc - 4, MAXFILES);
    for (i = 0; i < g_nfiles; i++)
    {
        file[i].f_name = argv[i + 4];
        file[i].f_host = argv[2];
        file[i].f_fd = -1;
        file[i].f_flags = 0;
    }
    printf("g_nfiles = %d\n", g_nfiles);

    home_page(argv[2], argv[3]);

    FD_ZERO(&g_rset);
    FD_ZERO(&g_wset);

    g_nconn = 0;
    g_maxfd = -1;
    g_nlefttoread = g_nlefttoconn = g_nfiles;

    while (g_nlefttoread > 0)
    {
        while (g_nconn < max_nconn && g_nlefttoconn > 0)
        {
            for (i = 0 ; i < g_nfiles; i++)
            {
                if (file[i].f_flags == 0)
                    break;
            }

            if (i == g_nfiles)
                err_quit("g_nlefttoconn = %d but nothing found", g_nlefttoconn);

            start_connect(&file[i]);
            g_nconn++;
            g_nlefttoconn--;
        }

        fd_set rs = g_rset;
        fd_set ws = g_wset;
        int n = Select(g_maxfd+1, &rs, &ws, NULL, NULL);

        for (i = 0; i < g_nfiles; i++)
        {
            int flags = file[i].f_flags;
            if ((flags == 0) || (flags & F_DONE))
                continue;

            int fd = file[i].f_fd;
            if ((flags & F_CONNECTING) && (FD_ISSET(fd, &rs) || FD_ISSET(fd, &ws)))
            {
                int error;
                n = sizeof(error);
                if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, (socklen_t*)&n) < 0 ||
                    error != 0) {
                    err_ret("nonblocking connect failed for %s", file[i].f_name);
                }
                printf("connection established for %s\n", file[i].f_name);
                FD_CLR(fd, &g_wset);      /* no more writeability test */
                write_get_cmd(&file[i]);/* write() the GET command */
            }
            else if ((flags & F_READING) && FD_ISSET(fd, &rs))
            {
                n = Read(fd, buf, sizeof(buf));
                if (n == 0)
                {
                    printf("end-of-file on %s\n", file[i].f_name);
                    Close(fd);
                    file[i].f_flags = F_DONE;   /* clears F_READING */
                    FD_CLR(fd, &g_rset);
                    g_nconn--;
                    g_nlefttoread--;
                }
                else
                {
                    printf("read %d bytes from %s\n", n, file[i].f_name);
                }
            }
        }
    }
    exit(0);
}
