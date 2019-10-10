.PHONY: all clean

CPPFLAGS += -I.. -I../include -DHAVE_MSGHDR_MSG_CONTROL
CFLAGS += -g -Wall -Wpedantic -pthread

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/unix_api_wrapper.c \
				$(COMM)/error.c $(COMM)/stdio_wrapper.c $(COMM)/readn.c $(COMM)/tcp_helpers.c \
				$(COMM)/pthread_wrapper.c $(COMM)/read_fd.c $(COMM)/write_fd.c

CLIENTS = client
SERVERS = serv00 serv01 serv02 serv021 serv03 serv04 serv05
BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

clients: $(CLIENTS)

servers: $(SERVERS)

client: client.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

serv00: serv00.c pr_cpu_time.c web_child.c $(BACKBONE)
serv01: serv01.c pr_cpu_time.c web_child.c $(BACKBONE)
serv02: serv02.c pr_cpu_time.c web_child.c child02.c $(BACKBONE)
serv021: serv02.c pr_cpu_time.c web_child.c child021.c $(BACKBONE)
serv03: serv03.c pr_cpu_time.c web_child.c child03.c lock_fcntl.c $(BACKBONE)
serv04: serv03.c pr_cpu_time.c web_child.c child03.c lock_thread.c $(BACKBONE)
serv05: serv05.c pr_cpu_time.c web_child.c child05.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
