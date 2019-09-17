.PHONY: all clean

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BINS = tcpcli01 tcpserv01 tcpserv02

all: $(BINS)

tcpcli01: tcpcli01.c str_cli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv01: tcpserv01.c str_echo.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv02: tcpserv02.c str_echo.c $(COMM)/signal.c $(COMM)/sig_child_waitpid.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
