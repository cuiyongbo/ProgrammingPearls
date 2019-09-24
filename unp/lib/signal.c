#include "unp.h"

sig_func_t Signal(int signo, sig_func_t func)
{
    struct sigaction act, oact;
    act.sa_handler = func;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;

    if(signo == SIGALRM)
    {
#if defined(SA_INTERRUPT)
        act.sa_flags |= SA_INTERRUPT;
#endif
    }
    else
    {
#if defined(SA_RESTART)
        act.sa_flags |= SA_RESTART;
#endif
    }

    if(sigaction(signo, &act, &oact) < 0)
        return SIG_ERR;
    else
        return oact.sa_handler;
}
