# Makefile for Cilk++ example.  See source code for details.

GPP	= g++
LIBARG	= -fopenmp
DEPENDENCIES = ../cpp/graph.cpp
TARGET1 = fw_iterative
TARGET2 = fw_recursive1
TARGET3 = fw_recursive2
TARGET4 = fw_recursive3
SRC1	= $(TARGET1).cpp $(DEPENDENCIES)
SRC2	= $(TARGET2).cpp $(DEPENDENCIES)
SRC3	= $(TARGET3).cpp $(DEPENDENCIES)
SRC4	= $(TARGET4).cpp $(DEPENDENCIES)


all: $(TARGET2) $(TARGET3) $(TARGET4)

$(TARGET1): $(SRC1)
	$(GPP) $(SRC1) $(LIBARG) -o $@

$(TARGET2): $(SRC2)
	$(GPP) $(SRC2) $(LIBARG) -o $@

$(TARGET3): $(SRC3)
	$(GPP) $(SRC3) $(LIBARG) -o $@	

$(TARGET4): $(SRC4)
	$(GPP) $(SRC4) $(LIBARG) -o $@	

clean:
	rm -f $(TARGET2) $(TARGET3) $(TARGET4)
