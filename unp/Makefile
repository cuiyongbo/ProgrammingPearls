.PHONY: all clean

clean:
	@rm -rf *.o a.out *.gch *.dSYM
	@find . -type d -exec file \{\} | grep executable | grep -v script | cut -d: -f1 | xargs rm -f \{\} \;

