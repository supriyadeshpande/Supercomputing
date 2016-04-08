CPP=mpic++
CFLAGS=-L. -L$CILKHOME/lib64 -Wl,-rpath=. -lsort

all: libsort.so
        $(CPP) $(CFLAGS) dist_sort.cpp ../cpp/array.cpp -o dist_sort
libsort.so:
        icc -m64 -fPIC -shared -o libsort.so merge_sort_1.cpp ../cpp/array.cpp
clean:
        rm -r dist_sort


