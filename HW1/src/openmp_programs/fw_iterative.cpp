#include "omp.h"
//#include <cilkview.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "../headers/graph.h"
#define TRUE 1
#define NUM_THREADS 4

using namespace std;

//HAVEN'T PARALLELIZED YET.
void fw_iterative(int ** matrix, int n){
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

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    int m = 8;

    int ** x = NULL;
    long long start, end;
    x = generate_matrix(n);

    ofstream myfile;
    myfile.open("/work/04026/tg833255/submission/openmp_runtimes/iterative.txt", ios_base::app);
    start = clock();
    fw_iterative(x, n);
    end = clock();
    myfile << "N = " << n << " M = " << m << " TIME: " << (end-start)/double(1000) << endl;
    myfile.close();
    return 0;

}





