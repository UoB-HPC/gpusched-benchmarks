default: fib

MODEL=openmp

include $(MODEL)/makefile.flags

CFLAGS+=$(EXTRA_FLAGS)

fib: main.o fib.o
	$(CC) $^ -o $@ $(LIBS) 

main.o: main.cc
	$(CC) $(CFLAGS) $^ -c -o $@

fib.o: $(MODEL)/fib.cc
	$(CC) $(CFLAGS) $^ -c -o $@

clean:
	-rm -f main.o fib.o fib
