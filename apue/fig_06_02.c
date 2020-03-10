#include "apue.h"
#include <pwd.h>

struct passwd* getpwnam_1(const char* name)
{
	struct passwd* ptr = NULL;
	
	setpwent();
	while((ptr=getpwent()) != NULL)
	{
		if(strcmp(name, ptr->pw_name) == 0)
			break;
	}
	endpwent();
	return ptr;
}
