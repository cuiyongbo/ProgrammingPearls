.PHONY: all clean 

all:

clean:
	@rm -rf *.o *.dSYM a.out *.i *.s
	@file * | grep executable | grep -v script | cut -d: -f1 | xargs rm -f

