.PHONY: all clean client server

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
CLIENTS = tcpcli01 tcpcli04
SERVERS = tcpserv01 tcpserv02 tcpserv03 tcpserv04
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

client: $(CLIENTS)

server: $(SERVERS)

tcpcli01: tcpcli01.c str_cli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli04: tcpcli04.c str_cli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv01: tcpserv01.c str_echo.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv02: tcpserv02.c str_echo.c $(COMM)/signal.c $(COMM)/sig_child_waitpid.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv03: tcpserv03.c str_echo.c $(COMM)/signal.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv04: tcpserv04.c str_echo.c $(COMM)/signal.c $(COMM)/sig_child_waitpid.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
