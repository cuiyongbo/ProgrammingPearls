#include "apue.h"
#include <aio.h>

// compile with: gcc aio_demo_01.c -lrt

#define BUF_SIZE 20     /* Size of buffers for read operations */

/* Application-defined structure for tracking I/O requests */
struct IoRequest 
{      
    int           reqNum;
    int           status;
    struct aiocb *aiocbp;
};

/* On delivery of SIGQUIT, we attempt to cancel all outstanding I/O requests */
static volatile sig_atomic_t gotSIGQUIT = 0;

static void quitHandler(int sig)
{
    gotSIGQUIT = 1;
}

#define IO_SIGNAL SIGUSR1   /* Signal used to notify I/O completion */

/* Handler for I/O completion signal */
static void aioSigHandler(int sig, siginfo_t *si, void *ucontext)
{
    if (si->si_code == SI_ASYNCIO) 
    {
        char buff[128];
        struct IoRequest *ioReq = si->si_value.sival_ptr;
        sprintf(buff, "FD %d: I/O completion signal received\n", ioReq->aiocbp->aio_fildes);
        write(STDOUT_FILENO, buff, strlen(buff));

        /* The corresponding IoRequest structure would be available as
                struct IoRequest *ioReq = si->si_value.sival_ptr;
            and the file descriptor would then be available via
                ioReq->aiocbp->aio_fildes */
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) 
    {
        err_quit("Usage: %s <pathname> <pathname>...",argv[0]);
    }

    // total requests
    int numReqs = argc - 1;

    /* Allocate our arrays */
    struct IoRequest* ioList = calloc(numReqs, sizeof(struct IoRequest));
    if (ioList == NULL)
        err_sys("calloc");

    struct aiocb *aiocbList = calloc(numReqs, sizeof(struct aiocb));
    if (aiocbList == NULL)
        err_sys("calloc");

    /* Establish handlers for SIGQUIT and the I/O completion signal */

    struct sigaction sa;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = quitHandler;
    if (sigaction(SIGQUIT, &sa, NULL) == -1)
        err_sys("sigaction");

    sa.sa_flags = SA_RESTART | SA_SIGINFO;
    sa.sa_sigaction = aioSigHandler;
    if (sigaction(IO_SIGNAL, &sa, NULL) == -1)
        err_sys("sigaction");

    /* Open each file specified on the command line, and queue
        a read request on the resulting file descriptor */

    for (int j = 0; j < numReqs; j++) 
    {
        ioList[j].reqNum = j;
        ioList[j].status = EINPROGRESS;
        ioList[j].aiocbp = &aiocbList[j];

        ioList[j].aiocbp->aio_fildes = open(argv[j + 1], O_RDONLY);
        if (ioList[j].aiocbp->aio_fildes == -1)
            err_sys("open");

        printf("opened %s on descriptor %d\n", argv[j + 1], ioList[j].aiocbp->aio_fildes);
        ioList[j].aiocbp->aio_buf = malloc(BUF_SIZE);
        if (ioList[j].aiocbp->aio_buf == NULL)
            err_sys("malloc");

        ioList[j].aiocbp->aio_nbytes = BUF_SIZE;
        ioList[j].aiocbp->aio_reqprio = 0;
        ioList[j].aiocbp->aio_offset = 0;
        ioList[j].aiocbp->aio_sigevent.sigev_notify = SIGEV_SIGNAL; // send IO_SIGNAL to process when asynchronous I/O finished
        ioList[j].aiocbp->aio_sigevent.sigev_signo = IO_SIGNAL;
        ioList[j].aiocbp->aio_sigevent.sigev_value.sival_ptr = &ioList[j];

        if (aio_read(ioList[j].aiocbp) == -1)
            err_sys("aio_read");
    }

    /* Number of I/O requests still in progress */
    int openReqs = numReqs;

    /* Loop, monitoring status of I/O requests */
    while (openReqs > 0) 
    {
        sleep(3);       /* Delay between each monitoring step */

        if (gotSIGQUIT) 
        {
            /* On receipt of SIGQUIT, attempt to cancel each of the
                outstanding I/O requests, and display status returned
                from the cancellation requests */

            printf("got SIGQUIT; canceling I/O requests: \n");

            for (int j = 0; j < numReqs; j++) 
            {
                if (ioList[j].status == EINPROGRESS) 
                {
                    printf("    Request %d on descriptor %d:", j, ioList[j].aiocbp->aio_fildes);
                    int s = aio_cancel(ioList[j].aiocbp->aio_fildes, ioList[j].aiocbp);
                    if (s == AIO_CANCELED)
                        printf("I/O canceled\n");
                    else if (s == AIO_NOTCANCELED)
                        printf("I/O not canceled\n");
                    else if (s == AIO_ALLDONE)
                        printf("I/O all done\n");
                    else
                        err_msg("aio_cancel");
                }
            }

            gotSIGQUIT = 0;
        }

        /* Check the status of each I/O request that is still in progress */
        printf("aio_error():\n");
        for (int j = 0; j < numReqs; j++) 
        {
            if (ioList[j].status == EINPROGRESS) 
            {
                printf("    for request %d (descriptor %d): ", j, ioList[j].aiocbp->aio_fildes);
                ioList[j].status = aio_error(ioList[j].aiocbp);

                switch (ioList[j].status) 
                {
                case 0:
                    printf("I/O succeeded\n");
                    break;
                case EINPROGRESS:
                    printf("In progress\n");
                    break;
                case ECANCELED:
                    printf("Canceled\n");
                    break;
                default:
                    err_msg("aio_error");
                    break;
                }

                if (ioList[j].status != EINPROGRESS)
                    openReqs--;
            }
        }
    }

    printf("All I/O requests completed\n");

    /* Check status return of all I/O requests */
    printf("aio_return():\n");
    for (int j = 0; j < numReqs; j++) 
    {
        ssize_t s = aio_return(ioList[j].aiocbp);
        printf("    for request %d (descriptor %d): %zd\n",
                j, ioList[j].aiocbp->aio_fildes, s);
    }

    exit(EXIT_SUCCESS);
}
