.PHONY: all clean

CPPFLAGS += -I../include -L..
CFLAGS += -g -Wall -Wpedantic
LIBS = -lunp -pthread
BINS = backlog

all: $(BINS)

backlog: backlog.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $? -o $@ $(LIBS)

clean:
	@rm -rf *.o *.a *.dSYM $(BINS) a.out
