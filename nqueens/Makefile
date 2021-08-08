EXE=nqueens
default: $(EXE)

CC=g++
CFLAGS=-std=c++11 -O3 -Wall -fopenmp
LIBS=-fopenmp
MODEL=openmp

include $(MODEL)/makefile.flags

CFLAGS+=$(EXTRA_FLAGS)

$(EXE): main.o $(EXE).o
	$(CC) $^ -o $@ $(LIBS) 

main.o: main.cc
	$(CC) $(CFLAGS) $^ -c -o $@

$(EXE).o: $(MODEL)/$(EXE).cc
	$(CC) $(CFLAGS) $^ -c -o $@

clean:
	-rm -f main.o $(EXE).o $(EXE)
