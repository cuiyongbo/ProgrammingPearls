.PHONY: all clean client server

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/sock_wrapper.c \
				$(COMM)/stdio_wrapper.c $(COMM)/unix_api_wrapper.c $(COMM)/error.c

CLIENTS = daytimetcpcli daytimetcpcliv6
SERVERS = daytimetcpserv
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

client: $(CLIENTS)

server: $(SERVERS)

daytimetcpcli: daytimetcpcli.c $(BACKBONE)
daytimetcpcliv6: daytimetcpcliv6.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

daytimetcpserv: daytimetcpserv.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
