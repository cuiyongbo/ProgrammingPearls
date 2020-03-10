#include "apue.h"
#define COUNTER_FILE "counter_file"

static int increaseCounter(char* marker);

int main()
{
    TELL_WAIT();

    FILE* fp = fopen(COUNTER_FILE, "w");
    if (fp == NULL)
    {
        err_sys("fopen error");
    }
    fprintf(fp, "%d", 0);
    fclose(fp);

    pid_t pid = fork();
    if (pid < 0)
    {
        err_sys("fork error");
    }
    else if(pid == 0)
    {
        
        int counter = 0;
        pid_t ppid = getppid();

        do 
        {
            WAIT_PARENT();

            counter = increaseCounter("Child");

            TELL_PARENT(ppid);
        } while(counter < 100);
    }
    else
    {
        while(increaseCounter("Parent") < 100)
        {
            TELL_CHILD(pid);
            WAIT_CHILD();
        }
    }

    return 0;
}

int increaseCounter(char* marker)
{
    FILE* fp = fopen(COUNTER_FILE, "r");
    if (fp == NULL)
    {
        err_sys("fopen error");
    }

    char buffer[32];
    int n = fread(buffer, sizeof(char), sizeof(buffer), fp);
    if (n < 0)
    {
        err_sys("fread error");
    }
    buffer[n] = 0;
    fclose(fp);

    int counter = atoi(buffer);
    printf("%s counter: %d\n", marker, counter);

    fp = fopen(COUNTER_FILE, "w");
    if (fp == NULL)
    {
        err_sys("fopen error");
    }

    counter++;
    fprintf(fp, "%d", counter);
    fclose(fp);
    return counter;
}