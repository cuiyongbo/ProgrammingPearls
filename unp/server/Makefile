.PHONY: all clean

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/unix_api_wrapper.c \
				$(COMM)/error.c $(COMM)/stdio_wrapper.c $(COMM)/readn.c $(COMM)/tcp_helpers.c

CLIENTS = client
SERVERS = serv00 serv01
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

client: $(CLIENTS)

server: $(SERVERS)

client: client.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

serv00: serv00.c pr_cpu_time.c web_child.c $(BACKBONE)
serv01: serv01.c pr_cpu_time.c web_child.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
