.PHONY: all clean

CPPFLAGS += -I../include -DHAVE_MSGHDR_MSG_CONTROL -DHAVE_MKSTEMP
CFLAGS += -g -Wall -Wpedantic -pthread

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/unix_api_wrapper.c \
				$(COMM)/error.c $(COMM)/stdio_wrapper.c $(COMM)/readn.c $(COMM)/tcp_helpers.c \
				$(COMM)/pthread_wrapper.c $(COMM)/read_fd.c $(COMM)/write_fd.c

CLIENTS = client
SERVERS = serv00 serv01 serv02 serv021 serv03 serv031 \
			serv04 serv05 serv06 serv07 serv08 serv09

BINS = $(CLIENTS) $(SERVERS)

all: $(BINS)

clients: $(CLIENTS)

servers: $(SERVERS)

client: client.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# iterative server
serv00: serv00.c pr_cpu_time.c web_child.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# concurrent server, one fork per client
serv01: serv01.c pr_cpu_time.c web_child.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prefork, no locking around 'accept'
serv02: serv02.c pr_cpu_time.c web_child.c child02.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# select collions
serv021: serv02.c pr_cpu_time.c web_child.c child021.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prefork, file lock around 'accept', similar to Apache server
serv03: serv03.c pr_cpu_time.c web_child.c child03.c lock_fcntl.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prefork, file lock around 'accept', metered
serv031: serv031.c pr_cpu_time.c web_child.c child031.c lock_fcntl.c meter.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prefork, mutex around 'accept'
serv04: serv03.c pr_cpu_time.c web_child.c child03.c lock_thread.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prefork, descriptor passing to children, similar to NSCA server
serv05: serv05.c pr_cpu_time.c web_child.c child05.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# one thread per client
serv06: serv06.c pr_cpu_time.c web_child.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prethreaded with mutex locking around 'accept'
serv07: serv07.c pr_cpu_time.c web_child.c pthread07.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prethreaded without mutex locking around 'accept'
serv09: serv09.c pr_cpu_time.c web_child.c pthread09.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

# prethreaded with only main thread doing 'accept'
serv08: serv08.c pr_cpu_time.c web_child.c pthread08.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@



clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
