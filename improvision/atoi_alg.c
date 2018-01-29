#include <stdio.h>
#include <ctype.h>

//bool my_isspace(char ch)
//{
//    return std::isspace(static_cast<unsigned char>(ch));
//}
//
//bool my_isdigit(char ch)
//{
//    return std::isdigit(static_cast<unsigned char>(ch));
//}


int myAtoi(const char* str)
{
	const char* p = str;
	while(isspace(*p))
		++p;
	
	int resultNum = 0;
	while(isdigit(*p))
		resultNum = resultNum*10 + *p - '0';

	return resultNum;
}

int main()
{
	char str[256];
	while(scanf("%s", str) != EOF)
	{
		printf("atoi = %d, myAtoi = %d\n", atoi(str), myAtoi(str));
	}

	return 0;
}






