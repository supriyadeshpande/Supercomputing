#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include "../headers/graph.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_SHARED_MEM_PER_BLOCK 1024
#define S_MATRIX_SIZE 32

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
			int m, int submatrix_offset){

	//Change the start cells, of x, u ,v
	x_row_st += submatrix_offset * m;
	x_col_st += blockIdx.y * m;
	u_row_st += submatrix_offset * m;
	u_col_st += submatrix_offset * m;
	v_row_st += submatrix_offset * m;
	v_col_st += blockIdx.y * m;

	if(blockIdx.y != submatrix_offset){
		// int tx = threadIdx.x, ty = threadIdx.y;
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
	
	
}


__global__
void CloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m, int submatrix_offset){


	x_row_st += blockIdx.x * m;
	x_col_st += submatrix_offset * m;
	u_row_st += blockIdx.x * m;
	u_col_st += submatrix_offset * m;
	v_row_st += submatrix_offset * m;
	v_col_st += submatrix_offset * m;

	if(blockIdx.x != submatrix_offset){
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
		}//outer k loop

	}

}




__global__
void DloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m, int submatrix_offset){

	x_row_st += blockIdx.x * m;
	x_col_st += blockIdx.y * m;
	u_row_st += blockIdx.x * m;
	u_col_st += submatrix_offset * m;
	v_row_st += submatrix_offset * m;
	v_col_st += blockIdx.y * m;

	if(blockIdx.x != k && blockIdx.y != k){

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
		}//outer for.

	}

	
} 


//Figure 4 implementation : HW 5
void AFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){
	int r = tilesize[depth];
	if(r > n){
		//Execute base case
		AloopFW(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
	}
	else{
		int sub_size = n / r;

		//Question 2.
		if(subsize < tilesize[depth+1]){

			for(int k = 0; k < r; k++){
			
				int offset = k * subsize;

				AloopFW(d_x, x_row_st + offset, x_col_st + offset, u_row_st + offset, u_col_st + offset, v_row_st + offset, v_col_st + offset, sub_size);

				cudaDeviceSynchronize();

				//Update kth row submatrices and kth col submatrices in parallel
				
				int threadX = min(sub_size, THREADS_PER_BLOCK_X);
				int threadY = min(sub_size, THREADS_PER_BLOCK_Y);
				dim3 threadsPerBlock(threadX, threadY);
				dim3 blocksPerGrid_B(1, r);

				BloopFW<<<blocksPerGrid_B, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k);
				cudaDeviceSynchronize();


				dim3 blocksPerGrid_C(r, 1);
				CloopFW<<<blocksPerGrid_C, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k);
				cudaDeviceSynchronize();

				//update remaining submatrices
				dim3 blocksPerGrid_D(r, r);
				DloopFW<<<blocksPerGrid_D, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k);
				cudaDeviceSynchronize();
			}
		}
		else{
			for(int k = 0; k < r; k++){
				int offset = k*sub_size;
				AFW(d_x, x_row_st + offset, x_col_st + offset, u_row_st + offset, u_col_st + offset, v_row_st + offset, v_col_st + offset, sub_size, depth+1, tilesize);
			

				for(int j = 0; j < r; j++){
					if(j == k)
						continue;
					BFW(d_x, x_row_st + offset, x_col_st + j*sub_size, u_row_st + offset, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);
					CFW(d_x, x_row_st + j*sub_size, x_col_st + offset, u_row_st + j*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + offset, sub_size, depth+1, tilesize);
				}
				

				for(int i = 0; i < r; i++){
					if(i == k)
						continue;
					for(int j = 0; j < r; j++){
						if(j == k)
							continue;
						DFW(d_x, x_row_st + i*sub_size, x_col_st + j*sub_size, u_row_st + i*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);
					}
				}

			}//outer for
			
		}//else

		
	}
}//AFW 




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
	cudaError_t err = cudaMalloc(&dev_matrix, (n+1) * sizeof(int *));
	if(err != cudaSuccess){
		printf("Error allocating memory on device.");
		return NULL;
	}
	for(int i = 1; i <= n; i++){
		//printf("%x\n", &addr[i]);
		int * start;
		err = cudaMalloc(&start, (n+1)*sizeof(int));
		if(err != cudaSuccess){
			printf("Error allocating memory on device.");
			return NULL;
		}
		cudaMemcpy(dev_matrix+i, &start, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(start, host_matrix[i], (n+1) * sizeof(int), cudaMemcpyHostToDevice);
	}
	return dev_matrix;
}

int main(int argc, char * argv[])
{
	
	//Matrix
	int n = atoi(argv[1]);
	int ** matrix = generate_matrix(n);
	int ** dev_matrix = copy_matrix_to_device(matrix, n);
	
	if(dev_matrix == NULL)
		return 0;	

	// fw_iterative_outer(dev_matrix, n);
	if(n <= 32){
		printf("Original matrix: \n");
		print_matrix(matrix, n);
	}	
	
	
	long long start, end;
	
	start = clock();
	int tilesize[2] = {4, INT_MAX};
	// int tilesize[3] = {2, 2, INT_MAX};
	AFW(dev_matrix, 1, 1, 1, 1, 1, 1, n, 0, tilesize);
	end = clock();
	
	if(n <= 32){
		int ** new_matrix = copy_matrix_to_host(dev_matrix, n);
		printf("\nWith updated distances: \n");
		print_matrix(new_matrix, n);
	}
	cout << "Runtime: " << double(end-start)/double(CLOCKS_PER_SEC) << endl;
	
	return 0;
}






