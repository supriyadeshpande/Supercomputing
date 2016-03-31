
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../headers/graph.h"

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


int main(int argc, char * argv[]){
	int n = 2048;
	int ** matrix = NULL;
	int ** copy = NULL;
	clock_t start, end;
	matrix = generate_matrix(n);
	//copy = copy_matrix(matrix, n);
/*
	start = clock();
	fw_iterative_serial(matrix, n);
	end = clock();
	std::cout << "Serial execution time: " << (end - start)/CLOCKS_PER_SEC << std::endl; 
*/
	start = clock();
	fw_iterative_serial(matrix, n);
	end = clock();
	std::cout << "Serial execution time: " << float(end - start)/CLOCKS_PER_SEC << std::endl; 
}


