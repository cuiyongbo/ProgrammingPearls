.PHONY: all clean

CPPFLAGS += -I../include -L..
CFLAGS += -g -Wall -Wpedantic
LIBS = -lunp -pthread

CLIENTS = client
SERVERS = serv00 serv01 serv02 serv021 serv03 serv031 \
			serv04 serv05 serv051 serv06 serv07 serv08 serv09

all: $(CLIENTS) $(SERVERS)
clients: $(CLIENTS)
servers: $(SERVERS)

client: client.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# iterative server
serv00: serv00.c pr_cpu_time.c web_child.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# concurrent server, one fork per client
serv01: serv01.c pr_cpu_time.c web_child.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prefork, no locking around 'accept'
serv02: serv02.c pr_cpu_time.c web_child.c child02.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# select collions
serv021: serv02.c pr_cpu_time.c web_child.c child021.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prefork, file lock around 'accept', similar to Apache server
serv03: serv03.c pr_cpu_time.c web_child.c child03.c lock_fcntl.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prefork, file lock around 'accept', metered
serv031: serv031.c pr_cpu_time.c web_child.c child031.c lock_fcntl.c meter.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prefork, mutex around 'accept'
serv04: serv03.c pr_cpu_time.c web_child.c child03.c lock_thread.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prefork, descriptor passing to children, similar to NSCA server
serv05: serv05.c pr_cpu_time.c web_child.c child05.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prefork, descriptor passing to children, using unix domain socket datagram
serv051: serv051.c pr_cpu_time.c web_child.c child051.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# one thread per client
serv06: serv06.c pr_cpu_time.c web_child.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prethreaded with mutex locking around 'accept'
serv07: serv07.c pr_cpu_time.c web_child.c pthread07.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prethreaded without mutex locking around 'accept'
serv09: serv09.c pr_cpu_time.c web_child.c pthread09.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

# prethreaded with only main thread doing 'accept'
serv08: serv08.c pr_cpu_time.c web_child.c pthread08.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(CLIENTS) $(SERVERS) a.out
