CC = mpicc
CFLAGS ?=

.PHONY: clean

all: ask1a ask1b ask1c ask1d

ask1a: ask1a.c
	${CC} ${CFLAGS} ask1a.c -o ask1a

ask1b: ask1b.c
	${CC} ${CFLAGS} -fopenmp ask1b.c -o ask1b

ask1c: ask1c.c
	${CC} ${CFLAGS} -fopenmp ask1c.c -o ask1c

ask1d: ask1d.c
	${CC} ${CFLAGS} ask1d.c -o ask1d -fopenmp -L ${ZFP_PATH}/build/lib/ -I ${ZFP_PATH}/include/ -lzfp -lm

clean:
	rm -f ask1a
	rm -f ask1b
	rm -f ask1c
	rm -f ask1d