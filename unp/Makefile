.PHONY: all clean

clean:
	@rm -rf *.o a.out *.gch *.dSYM
	@find . -type f -exec file \{\} \; | grep -E "ELF|executable" | grep -v script | cut -d: -f1 | xargs rm -f
	@find . -type d -name "*.dSYM" -exec rm -rf \{\} \;
