#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main()
{
    if (setvbuf(stdout, NULL, _IONBF, 0) != 0)
    {
        perror("setvbuf error");
        exit(EXIT_FAILURE);
    }

    int counter = 0;
    while(1)
    {
        if (counter % 5 == 0)
        {
            time_t ts = time(NULL);
            struct tm* dt = localtime(&ts);
            printf("tm_sec: %d, date: %s", dt->tm_sec, ctime(&ts));
        }

        sleep(60);
        counter++;
    }
}

// struct tm 
// {
//     int tm_sec;    /* Seconds (0-60) */
//     int tm_min;    /* Minutes (0-59) */
//     int tm_hour;   /* Hours (0-23) */
//     int tm_mday;   /* Day of the month (1-31) */
//     int tm_mon;    /* Month (0-11) */
//     int tm_year;   /* Year - 1900 */
//     int tm_wday;   /* Day of the week (0-6, Sunday = 0) */
//     int tm_yday;   /* Day in the year (0-365, 1 Jan = 0) */
//     int tm_isdst;  /* Daylight saving time */
// };
