#include <stdio.h>

void translateToLiteralDate(int year, int day)
{
	static const int daysOfMonth[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
	static const char* months[] = { "January", "February", "March", "April", "May", "June", 
				"July", "August", "September", "October", "December", "November" };

	if (day > 366)
	{
		printf("%d: Day number should be no larger than 366!\n", day);
		return;
	}

	bool plusOfLeapYear = year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
	int daysLeft = day;
	for (int i = 0; i < element_of(daysOfMonth); ++i)
	{
		if (daysLeft <= daysOfMonth[i])
		{
			printf("year %d , %d%s: %s %d, %d\n", 
						year, day, day % 10 == 1 ? "st" :( day %10 ==2 ? "nd": "th"), months[i], daysLeft, year);
			break;
		}

		daysLeft = daysLeft - daysOfMonth[i] - (i == 1 ? plusOfLeapYear : 0);
	}
}


int main(void)
{
	translateToLiteralDate(2017, 100);
	translateToLiteralDate(2017, 200);
	translateToLiteralDate(2017, 300);

	return 0;
}

