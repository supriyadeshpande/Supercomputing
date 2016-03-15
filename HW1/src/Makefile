# Makefile for Cilk++ example.  See source code for details.

CILKPP	= cilk++
LIBARG	= -O1 -g -lcilkutil
DEPENDENCIES = cpp/graph.cpp
TARGET1 = fw_iterative
TARGET2	= fw_recursive1
SRC1	= $(TARGET1).cilk $(DEPENDENCIES)
SRC2	= $(TARGET2).cilk $(DEPENDENCIES)	

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(SRC1)
	$(CILKPP) $(SRC1) $(LIBARG) -o $@

$(TARGET2): $(SRC2)
	$(CILKPP) $(SRC2) $(LIBARG) -o $@

clean:
	rm -f $(TARGET1) $(TARGET2)
