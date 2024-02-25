#include    "unp.h"

#define MAXFILES    20
#define SERV        "80"    /* port number or service name */

struct File
{
  char*  f_name;            /* filename */
  char*  f_host;            /* hostname or IPv4/IPv6 address */
  int    f_fd;              /* descriptor */
  int    f_flags;           /* F_xxx below */
} file[MAXFILES];

#define F_CONNECTING    1   /* connect() in progress */
#define F_READING       2   /* connect() complete; now reading */
#define F_DONE          4   /* all done */

#define GET_CMD     "GET %s HTTP/1.0\r\n\r\n"

int     g_nconn, g_nfiles, g_nlefttoconn, g_nlefttoread, g_maxfd;

fd_set  g_rset, g_wset;

void    home_page(const char *, const char *);
void    start_connect(struct File *);
void    write_get_cmd(struct File *);
