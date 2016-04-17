#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../headers/graph.h"

using namespace std;


__global__
void vecAdd(int ** x, int n){
	int i = threadIdx.x;
	if(i == 0){
		for(int i=1; i <= n; i++)
			for(int j = 1; j <= n; j++)
				x[i][j] += 1;		
	}
}

int ** copy_matrix_to_host(int ** dev_matrix, int n){
	int ** new_matrix = new int*[n+1];
	for(int i=1;i <= n; i++){
		new_matrix[i] = new int[n+1];
		int * begin;
		cudaMemcpy(&begin, &dev_matrix[i], sizeof (int *), cudaMemcpyDeviceToHost);
		printf("Here 1\n");
		cudaMemcpy(new_matrix[i], begin, (n+1) * sizeof(int), cudaMemcpyDeviceToHost);
	}
	return new_matrix;

}
int ** copy_matrix_to_device(int ** host_matrix, int n){
	//int ** dev_matrix = new int*[n+1];
	int ** addr;
	cudaMalloc(&addr, (n+1) * sizeof(int *));
	//printf("%x\n", addr);
	for(int i = 1; i <= n; i++){
		//printf("%x\n", &addr[i]);
		int * dev_mem;
		cudaMalloc(&dev_mem, (n+1)*sizeof(int));
		cudaMemcpy(addr+i, &dev_mem, sizeof(int *), cudaMemcpyHostToDevice);
		printf("here\n");
		//printf("%x %x\n", &dev_mem, dev_mem);
		cudaMemcpy(dev_mem, host_matrix[i], (n+1) * sizeof(int), cudaMemcpyHostToDevice);
		//dev_matrix[i] = new int[n+1];
		//cudaMalloc(&dev_matrix[i], (n+1) * sizeof(int));
		//cudaMemcpy(dev_matrix[i], host_matrix[i], n * sizeof(int), cudaMemcpyHostToDevice);
	}
	return addr;
}

int main(void)
{
	/*	
	//Test
	int n = 4;
	int  * x = {1,2,3,4};
	int *d_x;
	cudaMalloc(&d_x, n * sizeof(int));
	cudaMemcpy(d_x, x, n * sizeof(int), cudaMemcpyHostToDevice);
	vecAdd<<<1, 4>>>(d_x);
	cudaMemcpy(x, d_x, n*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < n; i++)
		cout << x[i] << " ";
	cout << endl;
	*/
	//Matrix
	int n = 4;
	int ** matrix = generate_matrix(n);
	int ** dev_matrix = copy_matrix_to_device(matrix, n);
	vecAdd<<<1,1>>>(dev_matrix, n);
	int ** new_matrix = copy_matrix_to_host(dev_matrix, n);
	printf("Old: \n");	
	print_matrix(matrix, n);
	printf("New: \n");
	print_matrix(new_matrix, n);
}





