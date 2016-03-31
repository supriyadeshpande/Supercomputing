#include "omp.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include "../headers/graph.h"
#define TRUE 1
#define NUM_THREADS 4

using namespace std;

int min(int a, int b){ return a < b ? a : b;}

void loopFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	int i,j,k; 
	for(k = 0; k < m; k++){
		//cout << "k=" << k << endl;
		for(i = 0; i < m; i++){
			for(j = 0; j < m; j++){
				x[x_row_st + i][x_col_st + j] = min( x[x_row_st + i][x_col_st + j], 
													(x[u_row_st + i][u_col_st + k] + x[v_row_st + k][v_col_st + j]) );
			}
		}
	}

}
/*
//Recursive implementation (PARALLEL)
void AFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

	//Incase of wrong values entered at runtime.
	if(m > n)
		return;

	//Recursion base case
	if(n == m)
		loopFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
	else{
		int mid = n/2;

		//AFW (X11, U11, V11)
		AFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);		

		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					//AFW (X12, U11, V12)
					AFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);		
				}
				#pragma omp section
				{
					//AFW (X21, U21, V11)
					AFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);	
				}
			}
		}

		//AFW (X22, U21, V12)
		AFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);

		//AFW (X22, U22, V22)
		AFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);


		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					//AFW (X21, U22, V21)
					AFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);		
				}
				#pragma omp section
				{
					//AFW (X12, U12, V22)
					AFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);		
				}	
			}			
		}		

		//AFW (X11, U12, V21)
		AFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);

	}

}
*/

/*
//USING TASK
//Recursive implementation (PARALLEL)
void AFW_parallel(int **x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

	//Incase of wrong values entered at runtime.
	if(m > n)
		return;

	//Recursion base case
	if(n == m)
		loopFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
	else{
		int mid = n/2;

		//AFW (X11, U11, V11)
		AFW_parallel(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);		

		#pragma omp parallel
		{
			#pragma omp single nowait
			{

				#pragma omp task
				AFW_parallel(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);
		
				#pragma omp task 
				AFW_parallel(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);	
				
				//#pragma omp taskwait
			}
			#pragma omp taskwait
		}
		
		single
		//AFW (X22, U21, V12)
		AFW_parallel(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);

		single
		//AFW (X22, U22, V22)
		AFW_parallel(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);


		#pragma omp parallel
		{
			#pragma omp single nowait
			{
				#pragma omp task 
				AFW_parallel(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);		
		
				#pragma omp task 
				AFW_parallel(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);		

				//#pragma omp taskwait
			}
			#pragma omp taskwait
		}	
		//AFW (X11, U12, V21)
		AFW_parallel(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);

	}

}

*/

void AFW_parallel(int **x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

	if(m > n)
		return;

	//Recursion base case
	if(n == m)
		loopFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
	else{
		int mid = n/2;

		//AFW (X11, U11, V11)
		AFW_parallel(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);		

		#pragma omp task
		AFW_parallel(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);

		#pragma omp task 
		AFW_parallel(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);	
		
		#pragma omp taskwait	

		//AFW (X22, U21, V12)
		AFW_parallel(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);

		//AFW (X22, U22, V22)
		AFW_parallel(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);

		#pragma omp task 
		AFW_parallel(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);		

		#pragma omp task 
		AFW_parallel(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);		

		#pragma omp taskwait

		//AFW (X11, U12, V21)
		AFW_parallel(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);

	}


}

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int m = 2;
    struct timeval start, end;
    int ** x = NULL;

    x = generate_matrix(n);

    //ofstream myfile;
    //myfile.open("/work/04026/tg833255/submission/openmp_runtimes/recursive2.txt", ios_base::app);
    //print_matrix(x, n);
    gettimeofday(&start, NULL);
    AFW_parallel(x, 1, 1, 1, 1, 1, 1, n, m);
   	gettimeofday(&end, NULL);
   	long duration = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
   	
   	//cout << "OUTPUT: " << endl;
   	//print_matrix(x, n);
   	cout << duration/1000000.0 << endl;

    //ofstream myfile;
    //myfile.open("/work/04026/tg833255/submission/openmp_runtimes/recursive1.txt", ios_base::app);
    //myfile << "N = " << n << " M = " << m << " TIME: " << (end-start)/double(1000) << endl;
    //myfile.close();
    return 0;

}





