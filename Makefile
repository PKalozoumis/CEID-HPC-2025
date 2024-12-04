.SILENT:

CC = mpicc
CFLAGS ?= 

.PHONY: clean

all: ask1a ask1b

ask1a: ask1a.c
	${CC} ${CFLAGS} ask1a.c -o ask1a

ask1b: ask1b.c
	${CC} ${CFLAGS} ask1b.c -o ask1b

clean:
	rm ask1a
	rm ask1b