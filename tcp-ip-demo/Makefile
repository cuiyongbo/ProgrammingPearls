.PHONY: all clean

CFLAGS = -g -Wall -Wextra

PROGS = udp_cli udp_serv tcp_cli tcp_serv

all: $(PROGS)

udp_cli: udp_client.o error.o
	$(CC) $(CFLAGS) $? -o $@

udp_serv: udp_server.o error.o
	$(CC) $(CFLAGS) $? -o $@

tcp_cli: tcp_client.o error.o read_stream.o
	$(CC) $(CFLAGS) $? -o $@

tcp_serv: tcp_server.o error.o read_stream.o
	$(CC) $(CFLAGS) $? -o $@


error.o: error.c
read_stream.o: read_stream.c cli_serv.h
tcp_client.o: tcp_client.c cli_serv.h
tcp_server.o: tcp_server.c cli_serv.h
udp_client.o: udp_client.c cli_serv.h
udp_server.o: udp_server.c cli_serv.h

clean:
	@rm -f *.o *.dSYM a.out $(PROGS)
	@file * | grep executable | grep -v script | cut -d: -f1 | xargs rm -f

