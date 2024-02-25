#include "apue.h"
#include <setjmp.h>

static jmp_buf env_alarm;

static void sig_alarm(int signo)
{
	// The longjmp() function uses the information saved in buffer `jmp_buf` to transfer
    // control back to the point where setjmp() was called and to restore
    // ("rewind") the stack to its state at the time of the setjmp() call.

    // Following a successful longjmp(), execution continues as if setjmp()
    // had returned for a second time. This "fake" return can be distin‐
    // guished from a true setjmp() call because the "fake" return returns
    // the value provided in val.

	longjmp(env_alarm, 1);
}

unsigned int sleep2(unsigned int seconds)
{
	if (signal(SIGALRM, sig_alarm) == SIG_ERR)
		return seconds;	

    // The  setjmp() function saves various information about the calling
    // environment (typically, the stack pointer, the instruction pointer,
    // possibly the values of other registers and the signal mask) in the
    // buffer `jmp_buf` for later use by longjmp(). In this case, setjmp() returns 0.

	if(setjmp(env_alarm) == 0)
	{
		alarm(seconds);
		pause();
	}
	return alarm(0);
}

static void sig_int(int signo)
{
	volatile int k = 0;
	printf("\nsig_int starting\n");
	for(int i=0; i<300000; i++)
		for(int j=0; j<4000; j++)
			k += i*j;
	printf("sig_int finished\n");
}

int main()
{
	if(signal(SIGINT, sig_int) == SIG_ERR)
	{
		err_sys("signal(SIGINT) error");
	}

	unsigned int unslept = sleep2(50);
	printf("sleep2 returned: %u\n", unslept);
	exit(0);
}

