.PHONY: all clean

CPPFLAGS += -I.. 

BINS = tcpcli01 tcpserv01

all: $(BINS)

tcpcli01: tcpcli01.c str_cli.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv01: tcpserv01.c str_echo.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
