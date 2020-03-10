.PHONY: all clean client server

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/unix_api_wrapper.c \
				$(COMM)/error.c $(COMM)/stdio_wrapper.c

CLIENTS = udpcli01
SERVERS = udpserv01
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

client: $(CLIENTS)

server: $(SERVERS)

udpcli01: udpcli01.c $(COMM)/dg_cli.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

udpserv01: udpserv01.c $(COMM)/dg_echo.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
