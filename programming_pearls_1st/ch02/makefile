.PHONY: all clean sign squash

CC = gcc
CFLAGS = -g -Wall

SIGN = proc_sign
SQUASH = proc_squash

all: $(SIGN) $(SQUASH)

clean:
	@rm -rf *.o *.dSYM *.pyc *.swap proc* *.out
	@file * | grep executable | grep -v script | cut -d: -f 1 | xargs rm

sign: $(SIGN)
squash: $(SQUASH)

sign.o: sign.c
proc_sign: sign.o
	$(CC) $(CFLAGS) $^ -o $@

squash.o: squash.c
proc_squash: squash.o
	$(CC) $(CFLAGS) $^ -o $@

