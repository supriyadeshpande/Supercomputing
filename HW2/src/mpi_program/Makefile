# Makefile for Cilk++ example.  See source code for details.

CC	= mpic++
CILKPP = cilk++
LIBARG	= -O1 -g -lcilkutil
DEPENDENCIES = ../cpp/array.cpp
TARGET1 = dist_sort
TARGET2 = merge_sort_1
SRC1	= $(TARGET1).cpp $(DEPENDENCIES)
SRC2 	= $(TARGET2).cilk $(DEPENDENCIES)
all: $(TARGET1) $(TARGET2)

$(TARGET1): $(SRC1)
	$(CC) $(SRC1) -o $@

$(TARGET2): $(SRC2)
	$(CILKPP) $(SRC2) $(LIBARG) -o $@

clean:
	rm -f $(TARGET1)
