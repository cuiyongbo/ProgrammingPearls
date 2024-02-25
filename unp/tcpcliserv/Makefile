.PHONY: all clean client server

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/unix_api_wrapper.c \
				$(COMM)/error.c

CLIENTS = tcpcli01 tcpcli11 tcpcli04 tcpcli09 tsigpipe
SERVERS = tcpserv01 tcpserv02 tcpserv03 tcpserv04 tcpserv08 tcpserv09
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

client: $(CLIENTS)

server: $(SERVERS)

tcpcli01: tcpcli01.c $(COMM)/str_cli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli11: tcpcli11.c str_cli11.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli04: tcpcli04.c $(COMM)/str_cli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli09: tcpcli04.c str_cli09.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tsigpipe: tsigpipe.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv01: tcpserv01.c $(COMM)/str_echo.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv02: tcpserv02.c $(COMM)/str_echo.c $(COMM)/signal.c $(COMM)/sig_child_waitpid.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv03: tcpserv03.c $(COMM)/str_echo.c $(COMM)/signal.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv04: tcpserv04.c $(COMM)/str_echo.c $(COMM)/signal.c $(COMM)/sig_child_waitpid.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv08: tcpserv04.c str_echo08.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv09: tcpserv04.c str_echo09.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
