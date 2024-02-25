.PHONY: clean

clean:
	@find . -name "a.out" -delete
	@find . -name "*.o" -delete
	@find . -name "*.pyc" -delete
	@find . -name "*.gch" -delete
	@find . -name "*.dSYM" -delete
