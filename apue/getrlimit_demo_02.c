#include "apue.h"

int main()
{
	int getrlimit(int resource, struct rlimit *rlim);
	struct rlimit rlim;
	if(getrlimit(RLIMIT_NOFILE, &rlim) != 0)
	{
		err_sys("getrlimit error");
	}

	printf("soft limit: %d, hard limit: %d\n", (int)rlim.rlim_cur, (int)rlim.rlim_max);

	rlim.rlim_cur = rlim.rlim_max;

	if(setrlimit(RLIMIT_NOFILE, &rlim) != 0)
	{
		err_sys("setrlimit error");
	}

	if(getrlimit(RLIMIT_NOFILE, &rlim) != 0)
	{
		err_sys("getrlimit error");
	}

	printf("soft limit: %d, hard limit: %d\n", (int)rlim.rlim_cur, (int)rlim.rlim_max);

	return 0;
}
