#include    "unp.h"

char* gf_time(void)
{
    struct timeval  tv;
    if (gettimeofday(&tv, NULL) < 0)
        err_sys("gettimeofday error");

    time_t t = tv.tv_sec;  /* POSIX says tv.tv_sec is time_t; some BSDs don't agree. */
    char* ptr = ctime(&t);

    static char str[30];
    strcpy(str, &ptr[11]);
        /* Fri Sep 13 00:00:00 1986\n\0 */
        /* 0123456789012345678901234 5  */
    snprintf(str+8, sizeof(str)-8, ".%06ld", (long)tv.tv_usec);
    return(str);
}
