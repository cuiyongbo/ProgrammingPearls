#include    "unp.h"

static struct flock g_lock_it, g_unlock_it;
static int          g_lock_fd = -1;

void my_lock_init(char *pathname)
{
    /* must copy caller's string, in case it's a constant */
    char    lock_file[1024];
    strncpy(lock_file, pathname, sizeof(lock_file));
    g_lock_fd = Mkstemp(lock_file);

    Unlink(lock_file);          /* but g_lock_fd remains open */

    g_lock_it.l_type = F_WRLCK;
    g_lock_it.l_whence = SEEK_SET;
    g_lock_it.l_start = 0;
    g_lock_it.l_len = 0;

    g_unlock_it.l_type = F_UNLCK;
    g_unlock_it.l_whence = SEEK_SET;
    g_unlock_it.l_start = 0;
    g_unlock_it.l_len = 0;
}

void my_lock_wait()
{
    int     rc;
    while ((rc = fcntl(g_lock_fd, F_SETLKW, &g_lock_it)) < 0)
    {
        if (errno == EINTR)
            continue;
        else
            err_sys("fcntl error for my_lock_wait");
    }
}

void my_lock_release()
{
    if (fcntl(g_lock_fd, F_SETLKW, &g_unlock_it) < 0)
        err_sys("fcntl error for my_lock_release");
}
