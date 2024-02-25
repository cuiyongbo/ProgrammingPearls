#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

static FILE* g_default_fp = NULL;

void set_default_stream(FILE *fp)
{
    g_default_fp = fp;
}

int mprintf(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    if (g_default_fp == NULL)
        g_default_fp = stdout;

    int rv = vfprintf(g_default_fp, fmt, args);

    va_end(args);
    return(rv);
}

int main()
{
    FILE* fp1 = fopen("test.log", "a");
    set_default_stream(fp1);
    mprintf("hello world\n");
    mprintf("%d %f %s\n", 1, 3.1415926, "on fire");

    set_default_stream(stdout);

    fflush(fp1);
    fclose(fp1);

    mprintf("hello world\n");
    mprintf("%d %f %s\n", 1, 3.1415926, "on fire");
}