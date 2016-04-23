#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include "../headers/graph.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define MAX_THREADS_PER_BLOCK 1024


using namespace std;


__global__
void AloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	int tx = threadIdx.x, ty = threadIdx.y, sum;

	int rowsPerThread = m / blockDim.x;
	int colsPerThread = m / blockDim.y;

	int r_offset_start = threadIdx.x * rowsPerThread;
	int r_offset_end = r_offset_start + rowsPerThread - 1;

	int c_offset_start = threadIdx.y * colsPerThread;
	int c_offset_end = c_offset_start + colsPerThread - 1;

	for(int k = 0; k < m; k++){

		if(tx == 0 && ty == 0){
			//update cell (k,k)
			sum = d_x[u_row_st + k][u_col_st + k] + d_x[v_row_st + k][v_col_st + k];
			d_x[x_row_st + k][x_col_st + k] = d_x[x_row_st + k][x_col_st + k] > sum ? sum : d_x[x_row_st + k][x_col_st + k];	
		}
			
		syncthreads();

		//Thread X responsible for updating current row.
		if(r_offset_start <= k && k<= r_offset_end){
			for(int j = c_offset_start; j <= c_offset_end; j++){
				if(j == k)
					continue;
				sum = d_x[u_row_st + k][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
				if(d_x[x_row_st + k][x_col_st + j] > sum)
					d_x[x_row_st + k][x_col_st + j] = sum;
			}
		}

		syncthreads();

		//Thread Y responsible for updating current column
		if(c_offset_start <= k && k <= c_offset_end){
			for(int i = r_offset_start; i <= r_offset_end; i++){
				if(i == k)
					continue;
				sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + k];
				if(d_x[x_row_st + i][x_col_st + k] > sum)
					d_x[x_row_st + i][x_col_st + k] = sum;
			}
		}

		syncthreads();

		for(int i = r_offset_start; i <= r_offset_end; i++){
			if(i == k)
				continue;
			for(int j = c_offset_start; j <= c_offset_end; j++){
				if(j == k)
					continue;
				int sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
				if(d_x[x_row_st + i][x_col_st + j] > sum)
					d_x[x_row_st + i][x_col_st + j] = sum;
			}
		}
		syncthreads();

	}
	
}

__global__
void BloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	//int tx = threadIdx.x, ty = threadIdx.y;
	int sum;

	int rowsPerThread = m / blockDim.x;
	int colsPerThread = m / blockDim.y;

	int r_offset_start = threadIdx.x * rowsPerThread;
	int r_offset_end = r_offset_start + rowsPerThread - 1;

	int c_offset_start = threadIdx.y * colsPerThread;
	int c_offset_end = c_offset_start + colsPerThread - 1;	

	for(int k=0; k < m; k++){

		//Update kth row using the corresponding thread.
		if(r_offset_start <= k && k<= r_offset_end){
			for(int j = c_offset_start; j <= c_offset_end; j++){				
				sum = d_x[u_row_st + k][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
				if(d_x[x_row_st + k][x_col_st + j] > sum)
					d_x[x_row_st + k][x_col_st + j] = sum;
			}
		}

		syncthreads();

		//Update the other cells.
		for(int i = r_offset_start; i <= r_offset_end; i++){
			if(i == k)
				continue;
			for(int j = c_offset_start; j <= c_offset_end; j++){
				int sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
				if(d_x[x_row_st + i][x_col_st + j] > sum)
					d_x[x_row_st + i][x_col_st + j] = sum;
			}
		}

		syncthreads();
	}
}


__global__
void CloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	//int tx = threadIdx.x, ty = threadIdx.y;
	int sum;

	int rowsPerThread = m / blockDim.x;
	int colsPerThread = m / blockDim.y;

	int r_offset_start = threadIdx.x * rowsPerThread;
	int r_offset_end = r_offset_start + rowsPerThread - 1;

	int c_offset_start = threadIdx.y * colsPerThread;
	int c_offset_end = c_offset_start + colsPerThread - 1;	

	for(int k=0; k < m; k++){

		if(c_offset_start <= k && k <= c_offset_end){
			for(int i = r_offset_start; i <= r_offset_end; i++){
				sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + k];
				if(d_x[x_row_st + i][x_col_st + k] > sum)
					d_x[x_row_st + i][x_col_st + k] = sum;
			}
		}

		syncthreads();

		for(int i = r_offset_start; i <= r_offset_end; i++){
			for(int j = c_offset_start; j <= c_offset_end; j++){
				if(j == k)
					continue;
				int sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
				if(d_x[x_row_st + i][x_col_st + j] > sum)
					d_x[x_row_st + i][x_col_st + j] = sum;
			}
		}

		syncthreads();
	}	

}




__global__
void DloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	int rowsPerThread = m / blockDim.x;
	int colsPerThread = m / blockDim.y;

	int r_offset_start = threadIdx.x * rowsPerThread;
	int r_offset_end = r_offset_start + rowsPerThread - 1;

	int c_offset_start = threadIdx.y * colsPerThread;
	int c_offset_end = c_offset_start + colsPerThread - 1;
	
	for(int k = 0; k < m; k++){

		for(int i = r_offset_start; i <= r_offset_end; i++){
			for(int j = c_offset_start; j <= c_offset_end; j++){
				
				int sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
				if(d_x[x_row_st + i][x_col_st + j] > sum)
					d_x[x_row_st + i][x_col_st + j] = sum;
			}
		}
		syncthreads();
	}

} 

//Recursive-3 implementation in HW1

void DFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

		if(m > n)
			return;

		if(n == m){			
			
			int threadX = min(m, THREADS_PER_BLOCK_X);
			int threadY = min(m, THREADS_PER_BLOCK_Y);
			dim3 threadsPerBlock(threadX, threadY);

			DloopFW<<<1, threadsPerBlock>>>(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
		}
		else{
			int mid = n/2;
			//DFW (X11, U11, V11)
			DFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);
			
			//DFW (X12, U11, V12)
			DFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);
			
			//DFW (X21, U21, V11)
			DFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);

			//DFW (X22, U21, V12)
			DFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);

			//DFW (X11, U12, V21)
			DFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);

			//DFW (X12, U12, V22)
			DFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);

			//DFW (X21, U22, V21)
			DFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);

			//DFW (X22, U22, V22)
			DFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);
		}

		
}

void CFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

	if(m > n)
		return;

	if(n == m){		
		int threadX = min(m, THREADS_PER_BLOCK_X);
		int threadY = min(m, THREADS_PER_BLOCK_Y);
		dim3 threadsPerBlock(threadX, threadY);

		CloopFW<<<1, threadsPerBlock>>>(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
	}
	else{
		int mid = n/2;
		CFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);		
		CFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);

		DFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);
		DFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);

		CFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);
		CFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);

		DFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);	
		DFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
	}
}

void BFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

	if(m > n)
		return;

	if(n == m){
		int threadX = min(m, THREADS_PER_BLOCK_X);
		int threadY = min(m, THREADS_PER_BLOCK_Y);
		dim3 threadsPerBlock(threadX, threadY);

		BloopFW<<<1, threadsPerBlock>>>(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
	}
	else{
		int mid = n/2;
		BFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);
		BFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);

		DFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);
		DFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);

		BFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
		BFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);

		DFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
		DFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);
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
	if(n == m){
		int threadX = min(m, THREADS_PER_BLOCK_X);
		int threadY = min(m, THREADS_PER_BLOCK_Y);
		dim3 threadsPerBlock(threadX, threadY);

		AloopFW<<<1, threadsPerBlock>>>(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);	
	}
	else{
		int mid = n/2;

		//AFW (X11, U11, V11)
		AFW(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, mid, m);
		
		//BFW (X12, U11, V12)
		BFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st, v_row_st, v_col_st + mid, mid, m);

		//CFW (X21, U21, V11)
		CFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st, v_row_st, v_col_st, mid, m);

		//DFW (X22, U21, V12)
		DFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st, v_row_st, v_col_st + mid, mid, m);

		//AFW (X22, U22, V22)
		AFW(x, x_row_st + mid, x_col_st + mid, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);

		//BFW (X21, U22, V21)
		BFW(x, x_row_st + mid, x_col_st, u_row_st + mid, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);

		//CFW (X12, U12, V22)
		CFW(x, x_row_st, x_col_st + mid, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st + mid, mid, m);

		//DFW (X11, U12, V21)
		DFW(x, x_row_st, x_col_st, u_row_st, u_col_st + mid, v_row_st + mid, v_col_st, mid, m);
	}

}




int ** copy_matrix_to_host(int ** dev_matrix, int n){
	int ** new_matrix = new int*[n+1];
	for(int i=1;i <= n; i++){
		new_matrix[i] = new int[n+1];
		int * begin;
		cudaMemcpy(&begin, &dev_matrix[i], sizeof (int *), cudaMemcpyDeviceToHost);
		cudaMemcpy(new_matrix[i], begin, (n+1) * sizeof(int), cudaMemcpyDeviceToHost);
	}
	return new_matrix;
}

int ** copy_matrix_to_device(int ** host_matrix, int n){
	//int ** dev_matrix = new int*[n+1];
	int ** dev_matrix;
	cudaMalloc(&dev_matrix, (n+1) * sizeof(int *));
	for(int i = 1; i <= n; i++){
		//printf("%x\n", &addr[i]);
		int * start;
		cudaMalloc(&start, (n+1)*sizeof(int));
		cudaMemcpy(dev_matrix+i, &start, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(start, host_matrix[i], (n+1) * sizeof(int), cudaMemcpyHostToDevice);
	}
	return dev_matrix;
}

int main(int argc, char * argv[])
{
	
	//Matrix
	int n = atoi(argv[1]);
	int m = 4;
	int ** matrix = generate_matrix(n);
	int ** dev_matrix = copy_matrix_to_device(matrix, n);
	
	if(n <= 32){
		printf("Original matrix: \n");
		print_matrix(matrix, n);
	}

	long long start, end;
	start = clock();
	AFW(dev_matrix, 1, 1, 1, 1, 1, 1, n, m);
	end = clock();
	int ** new_matrix = copy_matrix_to_host(dev_matrix, n);
	if(n <= 32){
		printf("\nWith updated distances: \n");
		print_matrix(new_matrix, n);	
	}
	
	cout << "Runtime: " << double(end-start)/double(CLOCKS_PER_SEC) << endl;
	return 0;
}






