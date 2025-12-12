COMPILER=gcc
OPTIONS=-fPIC -shared -o
COMPILE=$(COMPILER) $(OPTIONS)

all: program

program: socket_transfer.c socket_transfer.h
	$(COMPILE) socket_transfer.so socket_transfer.c socket_transfer.h

clean:
	rm -f socket_transfer.so