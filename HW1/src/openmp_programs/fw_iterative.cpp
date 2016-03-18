#include "omp.h"
//#include <cilkview.h>
#include <iostream>
#include <ctime>
#include "../headers/graph.h"
#define TRUE 1
#define NUM_THREADS 4

using namespace std;

void fw_iterative_serial(int ** matrix, int n){
	int i,j,k = 0;

	for(k = 1; k <= n; k++){
		for(i = 1; i <= n; i++){
			for(j = 1; j <= n; j++){
				if(matrix[i][j] > matrix[i][k] + matrix[k][j])
					matrix[i][j] = matrix[i][k] + matrix[k][j];
			}
		}
	}
}

//Iterative FW using cilk_for
void fw_iterative_parallel(int ** matrix, int n){
	int k,i,j;
	/*
	if(!omp_get_nested())
		omp_set_nested(TRUE);
	*/
	//int ctr[10];
	//for(i=0; i < 10; i++)
	//	ctr[i] = 0;
	
	for(k = 1; k <= n; k++){
		//#pragma omp parallel
		//#pragma omp barrier
		#pragma omp parallel for		
		for(i = 1; i <= n; i++){
			//ctr[omp_get_thread_num()]++;
			//cout << omp_get_thread_num();
			for(j = 1; j <= n; j++){
				//cout << omp_get_thread_num();
				if(matrix[i][j] > matrix[i][k] + matrix[k][j])
					matrix[i][j] = matrix[i][k] + matrix[k][j];
			}
		}
	}

	//cout << ctr[0] << "\n" << ctr[1] << "\n" << ctr[2] << "\n" << ctr[3] << "\n" <<endl;  
}

void test(){
	#pragma omp parallel num_threads(NUM_THREADS)
	{
		int id = omp_get_thread_num();
		if(id == 1)
			cout << "foo\n" << endl;
		else
			cout << "bar" << endl;
	}
}

int main(int argc, char * argv[]){
	int n;
	test();
	
	/*
	int ** matrix =  NULL;
	int ** copy = NULL;
	long long start, end;
	cout << "Enter matrix size: \n";
	cin >> n;
	matrix = generate_matrix(n);
	copy = copy_matrix(matrix, n);

	start = clock();
	fw_iterative_serial(matrix, n);
	end = clock();
	cout << "Serial execution time: " << (end - start)/double(1000) << endl; 
	start = clock();
	fw_iterative_parallel(copy, n);
	end = clock();
	cout << "Parallel execution time: " << (end - start)/double(1000) << endl; 
	//print_matrix(copy, n);
	*/
}





