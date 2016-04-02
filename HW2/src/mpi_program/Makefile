# Makefile for Cilk++ example.  See source code for details.

CC	= mpic++
DEPENDENCIES = ../cpp/array.cpp
TARGET1 = dist_sort
SRC1	= $(TARGET1).cpp $(DEPENDENCIES)
all: $(TARGET1)

$(TARGET1): $(SRC1)
	$(CC) $(SRC1) -o $@


clean:
	rm -f $(TARGET1)
