#include "apue.h"

int main(int argc, char* argv[])
{
	if(argc != 4)
		err_quit("Usage: %s <pid> [<soft-limit> <hard-limit>]", argv[0]);

	pid_t pid = atoi(argv[1]);
	struct rlimit old, new;
    new.rlim_cur = atoi(argv[2]);
    new.rlim_max = atoi(argv[3]);

    if(prlimit(pid, RLIMIT_CPU, &new, &old) < 0)
        err_sys("prlimit error");

    printf("Previous limits: soft=%lld, hard=%lld\n", 
        (long long)old.rlim_cur, (long long)old.rlim_max);

    if(prlimit(pid, RLIMIT_CPU, NULL, &old) < 0)
        err_sys("prlimit error");

    printf("Current limits: soft=%lld, hard=%lld\n", 
        (long long)old.rlim_cur, (long long)old.rlim_max);

    return 0;
}
