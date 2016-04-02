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
    for(i = start; i <= end; i++){
        long currIndex = i;
        for(j = i-1; j >= 0; j--){
            if(x[j] > x[currIndex]){
                swap(x, j, currIndex);
                currIndex = j;
            }
            else
            	break;
        }
    }
}

void merge(double * x, int start, int mid, int end){
        if(end <= start)
                return;

        long leftSize = mid - start + 1;
        long rightSize = end - mid;
        double * left = new double[leftSize];
        double * right = new double[rightSize];

        memcpy(left, x + start, leftSize*sizeof(double));
        memcpy(right, x + mid + 1, rightSize*sizeof(double));

        long i=0, j=0, k=start;
        while(i < leftSize && j < rightSize){
                if(left[i] <= right[j])
                        x[k++] = left[i++];
                else
                        x[k++] = right[j++];
        }

        while(i < leftSize)
                x[k++] = left[i++];

        while(j < rightSize)
                x[k++] = right[j++];

        delete[] left;
        delete[] right;
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
        long temp, q1, q2, q3;
        long p3_copy = p3;


        if (n1 + n2 <= m2) {
        	long mid1 = floor((p1 + r1)/2);
                long mid2 = floor((p2 + r2)/2);

                //if(p1 <= r1)
                        merge(t, p1, mid1,  r1);

                        //insertion_sort(t, p1, r1);
                //if(p2 <= r2)
                        merge(t, p2, mid2,  r2);

                for(int i = p1; i < r2; i++)
                {
                        x[p3_copy++] = t[i];
                } 
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
		
        double *t;
		int i;
        long n = end - start + 1;

        if(n <= m3){
                insertion_sort(x, start, end);
                return;
        }

        if(n == 1)
                return;

        long mid = start + (end - start)/2;

        par_merge_sort_PM(x, start, mid, m2, m3, n_val);
        par_merge_sort_PM(x, mid+1, end, m2, m3, n_val);

        t = new double[n_val];
        for(i = 0; i < n_val; i++)
        	t[i] = x[i];

        par_merge(t, start, mid, mid + 1, end, x, start, m2);
        delete[] t;
}


int main(int argc, char * argv[])
{
        long n = 32;//long(pow(double(2), double(29)));
        long m2 = 8;// = atoi(argv[1]);
        long m3 = 4;

        double * x = generate_array(n);

        clock_t start, end;
        start = clock();
        par_merge_sort_PM(x, 0, n - 1, m2, m3, n);
        end = clock();
        long duration = double(end - start)/double(CLOCKS_PER_SEC);
        //printf("\n\n\n%ld", duration);

        for(int i = 0; i < n; i++)
            printf("%f\t", x[i]);

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
