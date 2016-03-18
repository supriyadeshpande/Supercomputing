#include "omp.h"
#include <iostream>
#include <ctime>
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

void CFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

	if(m > n)
		return;

	if(n == m)
		loopFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
	else{
		int mid = n/2;
		
		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					CFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);		
				}
				#pragma omp section
				{
					CFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);	
				}
			}
		}


		

		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					CFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);
				}
				#pragma omp section
				{
					CFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);	
				}
			}
		}

		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					CFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);
				}
				#pragma omp section
				{
					CFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);
				}
			}
		}


		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					CFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);	
				}
				#pragma omp section
				{
					CFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
				}
			}
		}
	}
}

void BFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

	if(m > n)
		return;

	if(n == m)
		loopFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
	else{
		int mid = n/2;

		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					BFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);
				}
				#pragma omp section
				{
					BFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);
				}
			}
		}

		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					BFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);
				}
				#pragma omp section
				{
					BFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);	
				}
			}
		}

		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					BFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
				}
				#pragma omp section
				{
					BFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);
				}
			}
		}


		#pragma omp parallel
		{
			#pragma omp sections
			{
				#pragma omp section
				{
					BFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
				}
				#pragma omp section
				{
					BFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);	
				}
			}
		}

	}
}

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
					//BFW (X12, U11, V12)
					BFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);
				}
				#pragma omp section
				{
					//CFW (X21, U21, V11)
					CFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);
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
					//BFW (X21, U22, V21)
					BFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
				}
				#pragma omp section
				{
					//CFW (X12, U12, V22)
					CFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);
				}
			}
		}
		
		//AFW (X11, U12, V21)
		AFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
	}

}



int main(int arc, char * argv[]){
	int n, m;
	int ** x = NULL;
	//int ** copy = NULL;
	long long start, end;
	cout << "Enter matrix size: \n";
	cin >> n;
	cout << "Enter base case size: \n";
	cin >> m;
	x = generate_matrix(n);
	start = clock();
	AFW(x, 1, 1, 1, 1, 1, 1, n, m);
	end = clock();
	std::cout << "Parallel execution time: " << (end - start)/double(1000) << std::endl; 
	//print_matrix(x, n);

	return 0;
}
