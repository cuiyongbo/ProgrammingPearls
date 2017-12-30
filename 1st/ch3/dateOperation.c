#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>

typedef struct DateInfo {
	int tm_year;	// years since 1900
	int tm_mon;	// months since January - [0, 11]
	int tm_mday;	// day of month - [1, 31]
	int tm_wday;	// days since Sunday - [0, 6]
	int tm_yday;	// days since January 1 - [0, 365]
} DateInfo;

int DateDifference(DateInfo from, DateInfo to);
int weekdayNumofDate(const DateInfo date);
void printCalendarOfDate(const DateInfo date);

void DateDifferenceParser(const char* from, const char* to);

void usage(const char* prog)
{
	printf("Usage: %s [options] date1 [date2]\n"
		"Option:\n"
		"-h Show help\n"
		"-d date1 date2 Compute the number of days from date1 to date2\n"
		"-c date Print the calendar of date\n"
		"-w date Print the weekday number of date\n"
		"Date format: yyyymmdd\n", prog);
}


// Date Format: YYYYmmdd, like 20171103, 20120504
DateInfo parseDateFromString(const char* dateStr)
{
	DateInfo input;
	int dateNum = atoi(dateStr);
	input.tm_mday = dateNum % 100;
	dateNum /= 100;
	input.tm_mon = dateNum % 100;
	dateNum /= 100;
	input.tm_year = dateNum;
	return input;
}

int isInvalidInput(const char* input)
{
	const char* p = input;
	while(isdigit(*p) != 0)
		++p;

	return (p-input)==8 && *p==0;
}

int main(int argc, char* argv[])
{
	int ch;
	while((ch = getopt(argc, argv, "d:c:w:h")) != -1) {
		switch(ch) {
			case 'h' :
			case ':' :
			case '?' :
			default :
				usage(argv[0]);
				break;
			case 'd' :
				if(argv[optind]==NULL || argv[optind-1]==NULL)
					usage(argv[0]);
				else
					DateDifferenceParser(argv[optind-1], argv[optind]);				
				break;
			case 'c':
				printf("%s -c %s\n", argv[0], argv[optind-1]);
				break;
			case 'w':
				printf("%s -w %s\n", argv[0], argv[optind-1]);
				break;
		}
	}

	return 0;
}

void DateDifferenceParser(const char* from, const char* to)
{
	// sanity check
	if(isInvalidInput(from)==0 || isInvalidInput(to)==0) {
		fprintf(stderr, "Invalid input(s)! DateFormat: yyyymmdd\n");
		return;
	}

	DateInfo fromDate = parseDateFromString(from);
	DateInfo toDate = parseDateFromString(to);
	
	int difference = DateDifference(fromDate, toDate);

	printf("%d\n", difference);
}


int DateDifference(DateInfo from, DateInfo to)
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


