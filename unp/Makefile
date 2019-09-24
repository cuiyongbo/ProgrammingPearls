.PHONY: all clean

clean:
	@rm -rf *.o a.out *.gch *.dSYM
	@find . -type f -exec file \{\} \; | grep ELF | grep -v script | cut -d: -f1 | xargs rm

