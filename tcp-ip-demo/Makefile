.PHONY: all clean

CFLAGS = -g -Wall -Wextra

PROGS = udp_cli udp_serv

all: $(PROGS)

udp_cli: udp_client.o error.o
	$(CC) $(CFLAGS) $? -o $@

udp_serv: udp_server.o error.o
	$(CC) $(CFLAGS) $? -o $@

error.o: error.c
read_stream.o: read_stream.c cli_serv.h
udp_client.o: udp_client.c cli_serv.h
udp_server.o: udp_server.c cli_serv.h

clean:
	@rm -f *.o *.dSYM a.out $(PROGS)
	@file * | grep executable | grep -v script | cut -d: -f1 | xargs rm -f

