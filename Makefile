.SILENT:

CC = mpicc
CFLAGS ?= 

.PHONY: clean

ask1: ask1.c
	${CC} ${CFLAGS} ask1.c -o ask1

clean:
	rm ask1