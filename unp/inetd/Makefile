.PHONY: all clean client server

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/sock_wrapper.c \
				$(COMM)/stdio_wrapper.c $(COMM)/unix_api_wrapper.c $(COMM)/error.c \
				$(COMM)/daemon_helpers.c $(COMM)/sock_ntop_host.c $(COMM)/tcp_helpers.c

CLIENTS = daytimetcpcli01
SERVERS = daytimetcpserv02 daytimetcpserv03
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)
client: $(CLIENTS)
server: $(SERVERS)

daytimetcpcli01: daytimetcpcli01.c $(BACKBONE)
daytimetcpserv02: daytimetcpserv02.c $(BACKBONE)
daytimetcpserv03: daytimetcpserv03.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
