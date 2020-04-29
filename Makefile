.PHONY: test clean

test:
	$(CC) ${CFLAGS} ${CPPFLAGS} test.c -o test

clean:
	@find . -name "a.out" -delete
	@find . -name "*.o" -delete
	@find . -name "*.pyc" -delete
	@find . -name "*.gch" -delete
