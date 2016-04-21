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
void AloopFW_inner(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m, int k){

	

	int row_offset = blockIdx.x*blockDim.x + threadIdx.x;
	int col_offset = blockIdx.y*blockDim.y + threadIdx.y;

	int sum = d_x[u_row_st + row_offset][u_col_st + k] + d_x[v_row_st + k][v_col_st + col_offset];
	if(d_x[x_row_st + row_offset][x_col_st + col_offset] > sum)
		d_x[x_row_st + row_offset][x_col_st + col_offset] = sum;
	

	
	/*
	int rowsPerThread = m / blockDim.x;
	int colsPerThread = m / blockDim.y;

	int r_offset_start = threadIdx.x * rowsPerThread;
	int r_offset_end = r_offset_start + rowsPerThread - 1;

	int c_offset_start = threadIdx.y * colsPerThread;
	int c_offset_end = c_offset_start + colsPerThread - 1;

	for(int i = r_offset_start; i <= r_offset_end; i++){
		for(int j = c_offset_start; j <= c_offset_end; j++){
			int sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
			if(d_x[x_row_st + i][x_col_st + j] > sum)
				d_x[x_row_st + i][x_col_st + j] = sum;
		}
	}
	*/
}

//Called from host (outermost for loop)
void AloopFW_outer(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	int i,j,k; 
	for(k = 0; k < m; k++){	

		int threadX = min(m, THREADS_PER_BLOCK_X);
		int threadY = min(m, THREADS_PER_BLOCK_Y);

		int blocksX = m % threadX == 0 ? m/threadX : m/threadX + 1;
		int blocksY = m % threadY == 0 ? m/threadY : m/threadY + 1;

		dim3 blocksPerGrid(blocksX, blocksY);
		dim3 threadsPerBlock(threadX, threadY);
		AloopFW_inner<<<blocksPerGrid, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m, k);
	}

}


__global__
void DloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	/*
	int kPerThread = m / blockDim.z;
	int rowsPerThread = m / blockDim.x;
	int colsPerThread = m / blockDim.y;

	int k_offset_start = threadIdx.z * kPerThread;
	int k_offset_end = k_offset_start + kPerThread - 1;

	int r_offset_start = threadIdx.x * rowsPerThread;
	int r_offset_end = r_offset_start + rowsPerThread - 1;

	int c_offset_start = threadIdx.y * colsPerThread;
	int c_offset_end = c_offset_start + colsPerThread - 1;

	for(int k = k_offset_start; k <= k_offset_end; k++){
		for(int i = r_offset_start; i <= r_offset_end; i++){
			for(int j = c_offset_start; j <= c_offset_end; j++){
				
				int sum = d_x[u_row_st + i][u_col_st + k] + d_x[v_row_st + k][v_col_st + j];
				if(d_x[x_row_st + i][x_col_st + j] > sum)
					d_x[x_row_st + i][x_col_st + j] = sum;
			}
		}
	}
	*/

	
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	
	int row_offset = blockIdx.x*blockDim.x + threadIdx.x;
	int col_offset = blockIdx.y*blockDim.y + threadIdx.y;

	int sum = d_x[u_row_st + row_offset][u_col_st + k] + d_x[v_row_st + k][v_col_st + col_offset];
	if(d_x[x_row_st + row_offset][x_col_st + col_offset] > sum)
		d_x[x_row_st + row_offset][x_col_st + col_offset] = sum;

} 

//Recursive-3 implementation in HW1

void DFW(int ** x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int m){

		if(m > n)
			return;

		if(n == m){		
			/*
			int threadZ = 1;
			int threadX = min(m, THREADS_PER_BLOCK_X);
			int threadY = min(m, THREADS_PER_BLOCK_Y);

			int blockZ = m;
			int blockX = m % threadX == 0 ? m/threadX : m/threadX + 1;
			int blockY = m % threadY == 0 ? m/threadY : m/threadY + 1;

			dim3 blocksPerGrid(blockX, blockY, blockZ);
			dim3 threadsPerBlock(threadX, threadY, threadZ);

			DloopFW<<<blocksPerGrid, threadsPerBlock>>>(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
			*/

			AloopFW_outer(x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, m);
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


void DFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){
	int r = tilesize[depth];
	if(r > n){
		//Execute base case
		AloopFW_outer(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
	}
	else
		printf("Here\n");

}


void CFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){
	int r = tilesize[depth];
	if(r > n){
		//Execute base case
		AloopFW_outer(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
	}
	else
		printf("Here\n");

}

void BFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){
	int r = tilesize[depth];
	if(r > n){
		//Execute base case
		AloopFW_outer(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
	}
	else
		printf("Here\n");
}



//Figure 4 implementation : HW 5
void AFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){
	int r = tilesize[depth];
	if(r > n){
		//Execute base case
		AloopFW_outer(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
	}
	else{
		int sub_size = n / r;
		for(int k = 0; k < r; k++){
			int offset = k*sub_size;
			AFW(d_x, x_row_st + offset, x_col_st + offset, u_row_st + offset, u_col_st + offset, v_row_st + offset, v_col_st + offset, sub_size, depth+1, tilesize);
			
			//SYNC POINT
			cudaDeviceSynchronize();
			for(int j = 0; j < r; j++){
				if(j == k)
					continue;
				BFW(d_x, x_row_st + offset, x_col_st + j*sub_size, u_row_st + offset, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);
				CFW(d_x, x_row_st + j*sub_size, x_col_st + offset, u_row_st + j*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + offset, sub_size, depth+1, tilesize);
			}
			cudaDeviceSynchronize();

			for(int i = 0; i < r; i++){
				if(i == k)
					continue;
				for(int j = 0; j < r; j++){
					if(j == k)
						continue;
					DFW(d_x, x_row_st + i*sub_size, x_col_st + j*sub_size, u_row_st + i*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);
				}
			}
		}
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
	int m = 4;
	int ** matrix = generate_matrix(n);
	int ** dev_matrix = copy_matrix_to_device(matrix, n);
	
	if(dev_matrix == NULL)
		return 0;	

	// fw_iterative_outer(dev_matrix, n);
	printf("Original matrix: \n");
	print_matrix(matrix, n);
	
	long long start, end;
	start = clock();
	int tilesize[2] = {4, INT_MAX};
	AFW(dev_matrix, 1, 1, 1, 1, 1, 1, n, 0, tilesize);
	end = clock();
	int ** new_matrix = copy_matrix_to_host(dev_matrix, n);
	printf("\nWith updated distances: \n");
	print_matrix(new_matrix, n);
	//cout << "Runtime: " << double(end-start)/double(CLOCKS_PER_SEC) << endl;
	return 0;
}






