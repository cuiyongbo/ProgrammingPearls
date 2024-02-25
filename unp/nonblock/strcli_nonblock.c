#include "unp.h"

void str_cli(FILE* fp, int sockFd)
{
    int val = Fcntl(sockFd, F_GETFL, 0);
    Fcntl(sockFd, F_SETFL, val|O_NONBLOCK);

    val = Fcntl(STDIN_FILENO, F_GETFL, 0);
    Fcntl(STDIN_FILENO, F_SETFL, val|O_NONBLOCK);

    val = Fcntl(STDOUT_FILENO, F_GETFL, 0);
    Fcntl(STDOUT_FILENO, F_SETFL, val|O_NONBLOCK);

    char to[MAXLINE], fr[MAXLINE];
    char *toiptr, *tooptr, *friptr, *froptr;
    toiptr = tooptr = to;
    friptr = froptr = fr;
    int stdin_eof = 0;

    fd_set rset, wset;
    ssize_t n, nwritten;
    int maxFdp = max(max(STDOUT_FILENO, STDIN_FILENO), sockFd) + 1;

    for (;;)
    {
        FD_ZERO(&rset);
        FD_ZERO(&wset);
        if(stdin_eof == 0 && toiptr != &to[MAXLINE])
            FD_SET(STDIN_FILENO, &rset);
        if(friptr != &fr[MAXLINE])
            FD_SET(sockFd, &rset);
        if(tooptr != toiptr)
            FD_SET(sockFd, &wset);
        if(froptr != friptr)
            FD_SET(STDOUT_FILENO, &wset);

        Select(maxFdp+1, &rset, &wset, NULL, NULL);

        if(FD_ISSET(STDIN_FILENO, &rset))
        {
            n = read(STDIN_FILENO, toiptr, &to[MAXLINE] - toiptr);
            if(n < 0)
            {
                if(errno == EWOULDBLOCK)
                    err_sys("read error on stdin");
            }
            else if(n == 0)
            {
                fprintf(stderr, "%s: EOF on stdin\n", gf_time());
                stdin_eof = 1;
                if(toiptr == tooptr)
                    Shutdown(sockFd, SHUT_WR);
            }
            else
            {
                fprintf(stderr, "%s: read %d bytes from stdin\n", gf_time(), (int)n);
                toiptr += n;
                FD_SET(sockFd, &wset);
            }
        }

        if(FD_ISSET(sockFd, &rset))
        {
            n = read(sockFd, friptr, &fr[MAXLINE] - friptr);
            if(n < 0)
            {
                if(errno == EWOULDBLOCK)
                    err_sys("read error on socket");
            }
            else if(n == 0)
            {
                fprintf(stderr, "%s: EOF on socket\n", gf_time());
                if (stdin_eof)
                    return;
                else
                    err_quit("str_cli: server terminated prematurely");
            }
            else
            {
                fprintf(stderr, "%s: read %d bytes from socket\n", gf_time(), (int)n);
                friptr += n;
                FD_SET(STDOUT_FILENO, &wset);
            }
        }

        if (FD_ISSET(STDOUT_FILENO, &wset) && ((n=friptr-froptr) > 0))
        {
            nwritten = write(STDOUT_FILENO, froptr, n);
            if(nwritten < 0)
            {
                if(errno != EWOULDBLOCK)
                    err_sys("write error to stdout");
            }
            else
            {
                fprintf(stderr, "%s: wrote %d bytes to stdout\n", gf_time(), (int)nwritten);
                froptr += nwritten;
                if(froptr == friptr)
                    froptr = friptr = fr;
            }
        }

        if (FD_ISSET(sockFd, &wset) && ((n=toiptr-tooptr) > 0))
        {
            nwritten = write(sockFd, tooptr, n);
            if(nwritten < 0)
            {
                if(errno != EWOULDBLOCK)
                    err_sys("write error to socket");
            }
            else
            {
                fprintf(stderr, "%s: wrote %d bytes to socket\n", gf_time(), (int)nwritten);
                tooptr += nwritten;
                if(tooptr == toiptr)
                {
                    tooptr = toiptr = to;
                    if(stdin_eof)
                        Shutdown(sockFd, SHUT_WR);
                }
            }
        }
    }
}
