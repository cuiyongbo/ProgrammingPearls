.PHONY: all clean client server web exercise

CPPFLAGS += -I..
CFLAGS += -g -Wall -Wpedantic

COMM = ../lib
BACKBONE = $(COMM)/sock_wrapper.c $(COMM)/read_line.c $(COMM)/write_wrapper.c \
				$(COMM)/signal.c $(COMM)/sig_child_waitpid.c $(COMM)/sock_wrapper.c \
				$(COMM)/stdio_wrapper.c $(COMM)/unix_api_wrapper.c $(COMM)/error.c \
				$(COMM)/daemon_helpers.c $(COMM)/sock_ntop_host.c $(COMM)/tcp_helpers.c \
				$(COMM)/gf_time.c $(COMM)/host_serv.c

CLIENTS = tcpcli01 tcpcli02 tcpcli03 tcpcli04 daytimetcpcli
SERVERS = tcpserv_select_01
WEB = web_client
WEB_SRC = web.c home_page.c start_connect.c write_get_cmd.c
EXERCISE = tcpcli_ex_03

BINS = $(CLIENTS) $(SERVERS) $(WEB) $(EXERCISE)

all: $(BINS)
client: $(CLIENTS)
server: $(SERVERS)
web: $(WEB)
exercise: $(EXERCISE)

daytimetcpcli: daytimetcpcli.c $(COMM)/connect_nonb.c $(BACKBONE)
tcpcli01: tcpcli01.c strcli_fork.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli02: tcpcli01.c strcli_nonblock.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli04: tcpcli01.c strcli_fork.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli03: tcpcli03.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

web_client: $(WEB_SRC) $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpserv_select_01: tcpserv_select_01.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

tcpcli_ex_03: tcpcli01.c strcli_fork_01.c $(BACKBONE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

clean:
	-rm -rf *.dSYM *.o *.s
	-rm -rf $(BINS) a.out
