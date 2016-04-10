#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <string.h>
#include "../headers/array.h"

using namespace std;


void print_array(double * x, long start, long end){
    // if(end < start)
    //     return;
    long i;
    for(i=start; i <= end; i++){
        cout << x[i] << " ";
    }
    cout << endl;
}


void insertion_sort(double * x, int start, int end){
	int i,j;
	if(start >= end)
		return;
	for(i = start+1; i <= end; i++){
		int currIndex = i;
		for(j = i-1; j >= start; j--){
			if(x[j] > x[currIndex]){
				swap(x, j, currIndex);
				currIndex = j;
			}
		}
	}
}


long binary_search_par(double key, double *arr, long start, long end){
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
            if(arr[mid] == key)
                return mid;
			end = mid - 1;
		}
	}
	return end+1;
}

//************************************************************************************************************************



void merge(double * t, long p1, long r1, long p2, long r2, double * x, long p3){
        // cout << "Input to merge: \n";
        long i=p1, j=p2, k=p3;
        while(i <= r1 && j <= r2){
            if(t[i] <= t[j]) {
                x[k++] = t[i++];
            }
            else {
                x[k++] = t[j++];
            }
        }

        while(i <= r1)  {
            x[k++] = t[i++];
        }

        while(j <= r2) {
            x[k++] = t[j++];
        }
}

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
                swap(&p1, &p2);
                swap(&r1, &r2);
                swap(&n1, &n2);
            }

            if(n1 == 0){
                // cout << "Returning here\n";
                return;
            }

            q1 = floor(( r1 + p1 ) / 2) ;
            q2 = binary_search_par(t[q1], t, p2, r2);
            q3 = p3 + (q1 - p1) + (q2 - p2);

            x[q3] = t[q1];

            cilk_spawn par_merge(t, p1, q1-1, p2, q2-1, x, p3, m2);
            par_merge(t, q1 + 1, r1, q2, r2, x, q3+1, m2);
            cilk_sync;
        }
        
}

void par_merge_sort_PM(double * x, long start, long end, long m2, long m3){

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

        cilk_spawn par_merge_sort_PM(x, start, mid, m2, m3);
        par_merge_sort_PM(x, mid+1, end, m2, m3);
        cilk_sync;

        long leftSize = mid - start + 1;


        for(i = 0; i < n; i++) {
                t[i] = x[start + i];
        }

        mid = mid - start + 1;

        par_merge(t, 0, mid-1, mid, n-1, x, start, m2);
        
        delete[] t;
}

int main(int argc, char * argv[]){
    long n = atoi(argv[1]);
    double *x = generate_array(n);
    long m2= 16;
    long m3=8;
    print_array(x, 0, n-1);
    par_merge_sort_PM(x, 0, n-1, m2, m3);
    // par_merge(t, 0, 2, 3, 4, x, 0, m2);
    cout << "After sorting: " << endl;
    print_array(x, 0, n-1);
}