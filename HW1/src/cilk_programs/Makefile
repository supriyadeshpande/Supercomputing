# Makefile for Cilk++ example.  See source code for details.

CILKPP	= cilk++
LIBARG	= -O1 -g -lcilkutil
DEPENDENCIES = ../cpp/graph.cpp
TARGET1 = fw_iterative
TARGET2	= fw_recursive1
TARGET3	= fw_recursive2
TARGET4	= fw_recursive3
TARGET5	= serial

SRC1	= $(TARGET1).cilk $(DEPENDENCIES)
SRC2	= $(TARGET2).cilk $(DEPENDENCIES)	
SRC3	= $(TARGET3).cilk $(DEPENDENCIES)
SRC4	= $(TARGET4).cilk $(DEPENDENCIES)
SRC5	= $(TARGET5).cpp $(DEPENDENCIES)

all: $(TARGET5)

$(TARGET1): $(SRC1)
	$(CILKPP) $(SRC1) $(LIBARG) -o $@

$(TARGET2): $(SRC2)
	$(CILKPP) $(SRC2) $(LIBARG) -o $@

$(TARGET3): $(SRC3)
	$(CILKPP) $(SRC3) $(LIBARG) -o $@	

$(TARGET4): $(SRC4)
	$(CILKPP) $(SRC4) $(LIBARG) -o $@	

$(TARGET5): $(SRC5)
	g++ $(SRC5) -o $@	

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5)
