.PHONY: all clean

all:
	@echo "Nothing to build"

clean:
	@rm -rf *.o *.dSYM *.pyc *.swap proc* *.out
	@file * | grep executable | grep -v 'shell script' | cut -d: -f 1 | xargs echo
