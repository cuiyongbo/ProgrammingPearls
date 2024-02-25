#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>

static const int daysOfMonth[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
static const char* weekDescription[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};

typedef struct DateInfo {
	int tm_year;	// years since 1900
	int tm_mon;	// months since January - [0, 11]
	int tm_mday;	// day of month - [1, 31]
	int tm_wday;	// week day num [0-6], Sunday - 0
} DateInfo;

inline int isLeapYear(int year);
int ordinalDayOfDate(DateInfo date);
int isLeapYear(int year) {
    return year % 400 == 0 || (year % 4 == 0 && year % 100 != 0); 
}

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

int dateDifferenceInDay(DateInfo from, DateInfo to);
int weekdayNumofDate(const DateInfo date);
void printCalendarOfDate(DateInfo date);

void dateDifferenceParser(const char* from, const char* to);
void weekdayNumParser(const char* input);
void calendarParser(const char* input);


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
					dateDifferenceParser(argv[optind-1], argv[optind]);				
				break;
			case 'c':
				calendarParser(argv[optind-1]);
				break;
			case 'w':
				weekdayNumParser(argv[optind-1]);
				break;
		}
	}

	return 0;
}

void dateDifferenceParser(const char* from, const char* to)
{
	// sanity check
	if(isInvalidInput(from)==0 || isInvalidInput(to)==0) {
		fprintf(stderr, "Invalid input(s)! DateFormat: yyyymmdd\n");
		return;
	}

	DateInfo fromDate = parseDateFromString(from);
	DateInfo toDate = parseDateFromString(to);
	
	int difference = dateDifferenceInDay(fromDate, toDate);

	printf("%d\n", difference);
}

int dateDifferenceInDay(DateInfo from, DateInfo to)
{
	int ordinalDayOfFrom = ordinalDayOfDate(from);
	int ordinalDayOfTo = ordinalDayOfDate(to);

	int differenceInDay = ordinalDayOfTo - ordinalDayOfFrom;

	for(int i=from.tm_year; i<to.tm_year; ++i)
		differenceInDay += isLeapYear(i) ? 366 : 365;

	return differenceInDay;
}

int ordinalDayOfDate(DateInfo dateOfInterest)
{
	int ordinalDay = 0;
	for(int i=0; i<dateOfInterest.tm_mon-1; ++i)
		ordinalDay += daysOfMonth[i];	

	ordinalDay += dateOfInterest.tm_mday;
	
	if(dateOfInterest.tm_mon>2) 
		ordinalDay += isLeapYear(dateOfInterest.tm_year);

	return ordinalDay;
}

void weekdayNumParser(const char* input)
{
	if(isInvalidInput(input)==0) {
		fprintf(stderr, "Invalid input(s)! DateFormat: yyyymmdd\n");
		return;
	}

	DateInfo inputDate = parseDateFromString(input);
	int weekdayNum = weekdayNumofDate(inputDate);
	
	printf("%s\n", weekDescription[weekdayNum]);
}

int weekdayNumofDate(const DateInfo date)
{
	DateInfo baseDate;
	baseDate.tm_year = 1970;
	baseDate.tm_mon = 1;
	baseDate.tm_mday = 1;

	static const int weekDayNumOfBaseDate = 4; // Thursday
	int differenceInDay = dateDifferenceInDay(baseDate, date);

	return (differenceInDay%7 + weekDayNumOfBaseDate) % 7;
}

void calendarParser(const char* input)
{
	if(isInvalidInput(input)==0) {
		fprintf(stderr, "Invalid input(s)! DateFormat: yyyymmdd\n");
		return;
	}

	DateInfo inputDate = parseDateFromString(input);
	printCalendarOfDate(inputDate);
}

void printCalendarOfDate(DateInfo inputDate)
{
	static const char* offsetOfWeekDay[] = {
		"", /* first day is Sunday*/ 
		"   \t",
		"   \t   \t",
		"   \t   \t   \t",
		"   \t   \t   \t   \t",
		"   \t   \t   \t   \t   \t",
		"   \t   \t   \t   \t   \t   \t", /* first day is Saturday*/
	};
	
	printf("\t\t\t%d-%d\t\t\t\n", inputDate.tm_year, inputDate.tm_mon);
	printf("Sun\tMon\tTue\tWed\tThu\tFri\tSat\n");

	int oldOrdinalDay = inputDate.tm_mday;
	
	inputDate.tm_mday = 1;
	int weekdayNum = weekdayNumofDate(inputDate); // the weekday number of the first day of the give month
	int dayNum = daysOfMonth[inputDate.tm_mon-1];
	
	printf("%s", offsetOfWeekDay[weekdayNum]); // print the offset of the first day
	
	for(int i=1; i<=dayNum; ++i) {
		if(i == oldOrdinalDay)
			printf("\033[1;32m%*d\033[0m", -3, i);
		else
			printf("%*d", -3, i);

		weekdayNum = (weekdayNum+1)%7;

		if(weekdayNum == 0)
			printf("\n");
		else
			printf("\t");
	}

	printf("\n");
}

