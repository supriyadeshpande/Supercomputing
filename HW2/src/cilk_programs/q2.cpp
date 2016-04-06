#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include "../headers/array.h"


using namespace std;

//Used for Base case: Sort numbers from x[start] to x[end]
void insertion_sort(double * x, long start, long end){
    long i,j;
    if(start >= end)
    	return;
    for(i = start+1; i <= end; i++){
        long currIndex = i;
        for(j = i-1; j >= start; j--){
            if(x[j] > x[currIndex]){
                swap(x, j, currIndex);
                currIndex = j;
            }
            else
            	break;
        }
    }
    

}


void merge(double * x, int p1, int r1, int p2, int r2, double * x_orig, int p3){
      
        long i=p1, j=p2, k=p3;
        while(i <= r1 && j <= r2){
                if(x[i] <= x[j])
                        x_orig[k++] = x[i++];
                else
                        x_orig[k++] = x[j++];
        }

        while(i <= r1)
                x_orig[k++] = x[i++];

        while(j <= r2)
                x_orig[k++] = x[j++];
}     


long binary_search(double key, double *arr, long start, long end)
{
	long mid;
	while(end >= start)
	{
		mid = (start + end)/2;
		if (arr[mid] < key)
		{
			start = mid + 1;	
		}
		else 
		{
			end = mid - 1;
		}
	}
	return end+1;
}

//Merge x[start] to x[mid], x[mid+1] to x[end]
void par_merge(double * t, long p1, long r1, long p2, long r2, double *x, long p3, long m2)
{
        long n1 = r1 - p1 + 1;
        long n2 = r2 - p2 + 1;
        long r3 = p3 + n1 + n2 - 1;
        long q1, q2, q3;
        long p3_copy = p3;

        if (n1 + n2 <= m2) {
    		merge(t, p1, r1, p2, r2, x, p3);             
        }
        else {
            if (n1 < n2)
            {
                swap(p1, p2);
                swap(r1, r2);
                swap(n1, n2);
            }

            if(n1 == 0 )
                return;

            q1 = floor(( r1 + p1 ) / 2) ;
            q2 = binary_search(t[q1], t, p2, r2);
            q3 = p3 + (q1 - p1) + (q2 - p2);

            x[q3] = t[q1];
            par_merge(t, p1, q1-1, p2, q2-1, x, p3, m2);
            par_merge(t, q1 + 1, r1, q2, r2, x, q3+1, m2);
        }

}


//Sort x[start] to x[end]
//NOTE: m is the base case size.
void par_merge_sort_PM(double * x, long start, long end, long m2, long m3, long n_val){
		
		int i;
        long n = end - start + 1;
		double * t = new double[n];
        
        if(n <= m3){
                insertion_sort(x, start, end);
                return;
        }

        if(n == 1)
                return;

        long mid = start + (end - start)/2;

        par_merge_sort_PM(x, start, mid, m2, m3, n_val);
        par_merge_sort_PM(x, mid+1, end, m2, m3, n_val);

        
        for(i = 0; i < n; i++)
        	t[i] = x[start + i];
        
        mid = n/2;
        par_merge(t, 0, mid-1, mid, n-1, x, start, m2);
        delete[] t;

}


int main(int argc, char * argv[])
{
        long n = 64;//long(pow(double(2), double(29)));
        long m2 = 4;// = atoi(argv[1]);
        long m3 = 2;

        double * x = generate_array(n);

        for(int i = 0; i < n; i++)
            printf("%ld ", (long)x[i]);
        cout << endl;
		clock_t start, end;
        start = clock();
        par_merge_sort_PM(x, 0, n - 1, m2, m3, n);
        end = clock();
        long duration = double(end - start)/double(CLOCKS_PER_SEC);
        //printf("\n\n\n%ld", duration);
		printf("%s", "After sorting\n");
        for(int i = 0; i < n; i++)
            printf("%ld ", (long)x[i]);

        cout << endl;
        //std::cout << "Serial execution time: " << (end - start)/CLOCKS_PER_SEC << std::endl;

        //ofstream myfile;
        //myfile.open("/work/04026/tg833255/HW2/submission/Q1.txt", ios_base::app);
        //myfile.open("/Users/supriyadeshpande/Desktop/Supercomputing - HW2/sol.txt");
        //myfile << "FOR 1 PROCESSOR, N: " << n << " TIME: " << duration << endl;
        //myfile.close();
        delete[] x;
        return 0;
}
