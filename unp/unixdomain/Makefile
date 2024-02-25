.PHONY: all clean

CPPFLAGS += -I../include -L..
CFLAGS += -g -Wall -Wpedantic -pthread

COMM = ../lib
LIBS = -lunp

CLIENTS = unixstrcli01 unixdgcli01 daytimetcpcli daytimeunixcli
SERVERS = unixstrserv01 unixstrserv02 unixdgserv01 daytimetcpserv01 \
			daytimeunixserv01 daytimeunixserv02 daytimetcpserv02
OTHERS = unix_bind openfile my_cat
BINS = $(CLIENTS) $(SERVERS) $(OTHERS)

all: $(BINS)
clients: $(CLIENTS)
servers: $(SERVERS)

unix_bind: unix_bind.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

unixstrcli01: unixstrcli01.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

unixstrserv01: unixstrserv01.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

unixstrserv02: unixstrserv01.c str_echo.c read_cred.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

unixdgcli01: unixdgcli01.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

unixdgserv01: unixdgserv01.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

openfile: openfile.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

my_cat: my_cat.c my_open.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

daytimetcpcli: daytimetcpcli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

daytimetcpserv01: daytimetcpserv01.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

daytimetcpserv02: daytimetcpserv02.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

daytimeunixcli: daytimeunixcli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

daytimeunixserv01: daytimeunixserv01.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

daytimeunixserv02: daytimeunixserv02.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
