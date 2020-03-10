.PHONY: all clean client server

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/unix_api_wrapper.c \
				$(COMM)/stdio_wrapper.c $(COMM)/tcp_helpers.c $(COMM)/error.c

CLIENTS = tcpsend01 tcpsend04 tcpsend05
SERVERS = tcprecv01 tcprecv02 tcprecv03 tcprecv04 tcprecv05 tcprecv03p
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

client: $(CLIENTS)

server: $(SERVERS)

tcpsend01: tcpsend01.c $(BACKBONE)
tcpsend04: tcpsend04.c $(BACKBONE)
tcpsend05: tcpsend05.c $(BACKBONE)
tcpsend06: tcpsend06.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcprecv01: tcprecv01.c $(BACKBONE)
tcprecv02: tcprecv02.c $(BACKBONE)
tcprecv03: tcprecv03.c $(BACKBONE)
tcprecv04: tcprecv04.c $(BACKBONE)
tcprecv05: tcprecv05.c $(BACKBONE)
tcprecv03p: exercise_24_02.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
