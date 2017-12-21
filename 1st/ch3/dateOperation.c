#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct DateInfo {
	int tm_year;	// years since 1900
	int tm_mon;	// months since January - [0, 11]
	int tm_mday;	// day of month - [1, 31]
	int tm_wday;	// days since Sunday - [0, 6]
	int tm_yday;	// days since January 1 - [0, 365]
};

int DateDifference(DateInfo from, DateInfo to);
int weekdayNumofDate(const DateInfo date);
void printCalendarOfDate(const DateInfo date);

void usage(const char* prog)
{
	printf("Usage: %s [options] date1 [date2]\n"
		"Option:\n"
		"-d date1 date2 Compute the number of days from date1 to date2\n"
		"-c date Print the calendar of date\n"
		"-w date Print the weekday number of date\n");
}

int main(int argc, char* argv)
{
	


	return 0;
}

int DateDiffenence(DateInfo from, DateInfo to)
{

	return 0;
}

int weekdayNumofDate(const DateInfo date)
{

	return 0;
}

void printCalendarOfDate(const DateInfo date)
{




}


