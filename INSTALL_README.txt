Steps:

1. Download cilk++ sdk from 
https://software.intel.com/en-us/articles/download-intel-cilk-sdk

2. Extract the folder (the extracted folder name will be cilk). Say the path is /home/cilk 

3. Add the following lines to ~/.bashrc 

export PATH=/home/cilk/bin:$PATH
export LD_LIBRARY_PATH=/home/cilk/lib:/home/aadarsh-ubuntu/Desktop/Supercomputing/cilk/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/cilk/lib:/home/cilk/lib64:$LIBRARY_PATH


4. Check if example program runs. CD into /cilk/examples/qsort

5. make

6. If everything goes well, there will be an executable created by the name 'qsort'

7. Run ./qsort and you should get an output.

ERRORS:


If errors: try sudo apt-get install g++-multilib

If you get error like:
/usr/bin/ld: cannot find crt1.o: No such file or directory cilk

Then add this line to ~/.bashrc 

export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH