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
#define AFW_CONST 1
#define BFW_CONST 2
#define CFW_CONST 3
#define DFW_CONST 4


using namespace std;

int ** copy_matrix_to_host(int ** dev_matrix, int n);

__global__
void AloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m){

	int tx = threadIdx.x, ty = threadIdx.y;
	int sum;

	int rowsPerThread = m / blockDim.x;
	int colsPerThread = m / blockDim.y;

	int r_offset_start = threadIdx.x * rowsPerThread;
	int r_offset_end = r_offset_start + rowsPerThread - 1;

	int c_offset_start = threadIdx.y * colsPerThread;
	int c_offset_end = c_offset_start + colsPerThread - 1;

	//Copy to shared memory
	__shared__ int s_x[S_MATRIX_SIZE][S_MATRIX_SIZE];
	s_x[tx][ty] = d_x[x_row_st + tx][x_col_st + ty];
	syncthreads();


	for(int k = 0; k < m; k++){

		for(int i = r_offset_start; i <= r_offset_end; i++){
			if(i == k)
				continue;
			for(int j = c_offset_start; j <= c_offset_end; j++){
				if(j == k)
					continue;
				
				sum = s_x[i][k] + s_x[k][j];
				if(s_x[i][j] > sum)
					s_x[i][j] = sum;
			}
		}
		syncthreads();

	}//outer k for loop

	syncthreads();
	d_x[x_row_st + tx][x_col_st + ty] = s_x[tx][ty];
	
}

__global__
void BloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m, int submatrix_offset, int parent){

	//Change the start cells, of x, u ,v
	x_row_st += submatrix_offset * m;
	x_col_st += blockIdx.y * m;
	u_row_st += submatrix_offset * m;
	u_col_st += submatrix_offset * m;
	v_row_st += submatrix_offset * m;
	v_col_st += blockIdx.y * m;

	if(!(blockIdx.y == submatrix_offset && parent == AFW_CONST)){
		int tx = threadIdx.x, ty = threadIdx.y;
		int sum;

		int rowsPerThread = m / blockDim.x;
		int colsPerThread = m / blockDim.y;

		int r_offset_start = threadIdx.x * rowsPerThread;
		int r_offset_end = r_offset_start + rowsPerThread - 1;

		int c_offset_start = threadIdx.y * colsPerThread;
		int c_offset_end = c_offset_start + colsPerThread - 1;	

		__shared__ int s_x[S_MATRIX_SIZE][S_MATRIX_SIZE];
		__shared__ int s_u[S_MATRIX_SIZE][S_MATRIX_SIZE];

		s_x[tx][ty] = d_x[x_row_st + tx][x_col_st + ty];
		s_u[tx][ty] = d_x[u_row_st + tx][u_col_st + ty];
		syncthreads();

		for(int k=0; k < m; k++){

			//Update the other cells.
			for(int i = r_offset_start; i <= r_offset_end; i++){
				if(i == k)
					continue;
				for(int j = c_offset_start; j <= c_offset_end; j++){
					sum = s_u[i][k] + s_x[k][j];
					if(s_x[i][j] > sum)
						s_x[i][j] = sum;	
				}
			}

			syncthreads();
		}//outer k for loop

		syncthreads();
		d_x[x_row_st + tx][x_col_st + ty] = s_x[tx][ty];
	}
	
	
}


__global__
void CloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m, int submatrix_offset, int parent){


	x_row_st += blockIdx.x * m;
	x_col_st += submatrix_offset * m;
	u_row_st += blockIdx.x * m;
	u_col_st += submatrix_offset * m;
	v_row_st += submatrix_offset * m;
	v_col_st += submatrix_offset * m;

	if(!(blockIdx.x == submatrix_offset && parent == AFW_CONST)){
		int tx = threadIdx.x, ty = threadIdx.y;
		int sum;

		int rowsPerThread = m / blockDim.x;
		int colsPerThread = m / blockDim.y;

		int r_offset_start = threadIdx.x * rowsPerThread;
		int r_offset_end = r_offset_start + rowsPerThread - 1;

		int c_offset_start = threadIdx.y * colsPerThread;
		int c_offset_end = c_offset_start + colsPerThread - 1;	

		__shared__ int s_x[S_MATRIX_SIZE][S_MATRIX_SIZE];
		__shared__ int s_v[S_MATRIX_SIZE][S_MATRIX_SIZE];

		s_x[tx][ty] = d_x[x_row_st + tx][x_col_st + ty];
		s_v[tx][ty] = d_x[v_row_st + tx][v_col_st + ty];
		syncthreads();

		for(int k=0; k < m; k++){

			for(int i = r_offset_start; i <= r_offset_end; i++){
				for(int j = c_offset_start; j <= c_offset_end; j++){
					if(j == k)
						continue;
					sum = s_x[i][k] + s_v[k][j];
					if(s_x[i][j] > sum)
						s_x[i][j] = sum;	
				}
			}

			syncthreads();
		}//outer k loop

		syncthreads();
		d_x[x_row_st + tx][x_col_st + ty] = s_x[tx][ty];

	}

}




__global__
void DloopFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int m, int submatrix_offset, int parent){

	x_row_st += blockIdx.x * m;
	x_col_st += blockIdx.y * m;
	u_row_st += blockIdx.x * m;
	u_col_st += submatrix_offset * m;
	v_row_st += submatrix_offset * m;
	v_col_st += blockIdx.y * m;


	//AFW_PARENT -> blockIdx.x != submatrix_offset && blockIdx.y != submatrix_offset
	//BFW_PARENT -> blockIdx.x != submatrix_offset
	//CFW_PARENT -> blockIdx.y != submatrix_offset
	int flag1 = parent == AFW_CONST && blockIdx.x == submatrix_offset && blockIdx.y == submatrix_offset;
	int flag2 = parent == BFW_CONST && blockIdx.x == submatrix_offset;
	int flag3 = parent == CFW_CONST && blockIdx.y == submatrix_offset; 

	if(!(flag1 || flag2 || flag3)){
		int tx = threadIdx.x, ty = threadIdx.y;
		int sum;

		int rowsPerThread = m / blockDim.x;
		int colsPerThread = m / blockDim.y;

		int r_offset_start = threadIdx.x * rowsPerThread;
		int r_offset_end = r_offset_start + rowsPerThread - 1;

		int c_offset_start = threadIdx.y * colsPerThread;
		int c_offset_end = c_offset_start + colsPerThread - 1;
		
		__shared__ int s_x[S_MATRIX_SIZE][S_MATRIX_SIZE];
		__shared__ int s_u[S_MATRIX_SIZE][S_MATRIX_SIZE];
		__shared__ int s_v[S_MATRIX_SIZE][S_MATRIX_SIZE];

		s_x[tx][ty] = d_x[x_row_st + tx][x_col_st + ty];
		s_u[tx][ty] = d_x[u_row_st + tx][u_col_st + ty];
		s_v[tx][ty] = d_x[v_row_st + tx][v_col_st + ty];
		syncthreads();

		for(int k = 0; k < m; k++){

			for(int i = r_offset_start; i <= r_offset_end; i++){
				for(int j = c_offset_start; j <= c_offset_end; j++){
					sum = s_u[i][k] + s_v[k][j];
					if(s_x[i][j] > sum)
						s_x[i][j] = sum;
				}
			}
			syncthreads();
		}//outer k for loop

		syncthreads();
		d_x[x_row_st + tx][x_col_st + ty] = s_x[tx][ty];
	}

	
} 


void DFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){
	int r = tilesize[depth];
	if(r > n){
		printf("ERR DFW: Shouldn't reach here.\n");
		/*
		int threadX = min(n, THREADS_PER_BLOCK_X);
		int threadY = min(n, THREADS_PER_BLOCK_Y);
		dim3 threadsPerBlock(threadX, threadY);
		//Execute base case
		DloopFW<<<1, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
		*/	
	}
	else{
		int sub_size = n / r;

		if(sub_size < tilesize[depth + 1]){
			int threadX = min(sub_size, THREADS_PER_BLOCK_X);
			int threadY = min(sub_size, THREADS_PER_BLOCK_Y);
			dim3 threadsPerBlock(threadX, threadY);	

			for(int k = 0; k < r; k++){

				//Update all submatrices with Dloop
				dim3 blocksPerGrid_D(r, r);
				DloopFW<<<blocksPerGrid_D, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, DFW_CONST);
				cudaDeviceSynchronize();
			}
		}
		else{

			for(int k = 0; k < r; k++){
				int offset = k*sub_size;
				for(int i = 0; i < r; i++){
					for(int j = 0; j < r; j++){
						DFW(d_x, x_row_st + i*sub_size, x_col_st + j*sub_size, u_row_st + i*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);
					}
				}
				cudaDeviceSynchronize();
			}//outer k loop
		}	

	}
}



void CFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){

	
	int r = tilesize[depth];
	if(r > n){
		printf("ERR CFW: Shouldn't reach here.\n");
		/*
		int threadX = min(n, THREADS_PER_BLOCK_X);
		int threadY = min(n, THREADS_PER_BLOCK_Y);
		dim3 threadsPerBlock(threadX, threadY);
		//Execute base case
		CloopFW<<<1, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
		*/	
	}
	else{
		int sub_size = n / r;

		if(sub_size < tilesize[depth + 1]){
			int threadX = min(sub_size, THREADS_PER_BLOCK_X);
			int threadY = min(sub_size, THREADS_PER_BLOCK_Y);
			dim3 threadsPerBlock(threadX, threadY);	

			for(int k = 0; k < r; k++){
				//Update kth col with Cloop
				dim3 blocksPerGrid_C(r, 1);
				CloopFW<<<blocksPerGrid_C, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, CFW_CONST);
				cudaDeviceSynchronize();

				//Update remaining cells with Dloop
				dim3 blocksPerGrid_D(r, r);
				DloopFW<<<blocksPerGrid_D, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, CFW_CONST);
				cudaDeviceSynchronize();

			}
		}
		else{

			for(int k = 0; k < r; k++){
				int offset = k*sub_size;

				for(int i = 0; i < r; i++){
					CFW(d_x, x_row_st + i*sub_size, x_col_st + offset, u_row_st + i*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + offset, sub_size, depth+1, tilesize);
				}

				for(int i = 0; i < r; i++){
					for(int j = 0; j < r; j++){
						if(j == k)
							continue;
						DFW(d_x, x_row_st + i*sub_size, x_col_st + j*sub_size, u_row_st + i*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);
					}
				}

			}//outer k loop
		}		

	}

}


void BFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){

	int r = tilesize[depth];
	if(r > n){
		printf("ERR BFW: Shouldn't reach here.\n");
		/*
		int threadX = min(n, THREADS_PER_BLOCK_X);
		int threadY = min(n, THREADS_PER_BLOCK_Y);
		dim3 threadsPerBlock(threadX, threadY);
		//Execute base case
		BloopFW<<<1, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
		*/	
	}
	else{
		int sub_size = n / r;

		if(sub_size < tilesize[depth + 1]){
			int threadX = min(sub_size, THREADS_PER_BLOCK_X);
			int threadY = min(sub_size, THREADS_PER_BLOCK_Y);
			dim3 threadsPerBlock(threadX, threadY);	

			for(int k = 0; k < r; k++){
				//Update kth row with Bloop
				dim3 blocksPerGrid_B(1, r);
				BloopFW<<<blocksPerGrid_B, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, BFW_CONST);
				cudaDeviceSynchronize();

				//Update remaining cells with Dloop
				dim3 blocksPerGrid_D(r, r);
				DloopFW<<<blocksPerGrid_D, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, BFW_CONST);
				cudaDeviceSynchronize();

			}
		}
		else{

			for(int k = 0; k < r; k++){
				int offset = k*sub_size;	

				for(int j = 0; j < r; j++){
					BFW(d_x, x_row_st + offset, x_col_st + j*sub_size, u_row_st + offset, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);			
				}

				for(int i = 0; i < r; i++){
					if(i == k)
						continue;
					for(int j = 0; j < r; j++){
						DFW(d_x, x_row_st + i*sub_size, x_col_st + j*sub_size, u_row_st + i*sub_size, u_col_st + offset, v_row_st + offset, v_col_st + j*sub_size, sub_size, depth+1, tilesize);
					}
				}

			}//outer k loop

		}		

	}
}

//Figure 4 implementation : HW 5
void AFW(int ** d_x, int x_row_st, int x_col_st, 
			int u_row_st, int u_col_st, 
			int v_row_st, int v_col_st,
			int n, int depth, int * tilesize){
	int r = tilesize[depth];
	if(r > n){
		printf("ERR AFW: Shouldn't reach here.\n");
		/*
		int threadX = min(n, THREADS_PER_BLOCK_X);
		int threadY = min(n, THREADS_PER_BLOCK_Y);
		dim3 threadsPerBlock(threadX, threadY);
		//Execute base case
		AloopFW<<<1, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, n);
		*/
	}
	else{
		int sub_size = n / r;
		
		if(sub_size < tilesize[depth+1]){
			
			int threadX = min(sub_size, THREADS_PER_BLOCK_X);
			int threadY = min(sub_size, THREADS_PER_BLOCK_Y);
			dim3 threadsPerBlock(threadX, threadY);

			for(int k = 0; k < r; k++){
			
				int offset = k * sub_size;

				AloopFW<<<1, threadsPerBlock>>>(d_x, x_row_st + offset, x_col_st + offset, u_row_st + offset, u_col_st + offset, v_row_st + offset, v_col_st + offset, sub_size);

				cudaDeviceSynchronize();

				//Update kth row submatrices and kth col submatrices in parallel
				
				dim3 blocksPerGrid_B(1, r);

				BloopFW<<<blocksPerGrid_B, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, AFW_CONST);
				cudaDeviceSynchronize();

				dim3 blocksPerGrid_C(r, 1);
				CloopFW<<<blocksPerGrid_C, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, AFW_CONST);
				cudaDeviceSynchronize();

				

				//update remaining submatrices
				dim3 blocksPerGrid_D(r, r);
				DloopFW<<<blocksPerGrid_D, threadsPerBlock>>>(d_x, x_row_st, x_col_st, u_row_st, u_col_st, v_row_st, v_col_st, sub_size, k, AFW_CONST);
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

}//end of iterative


int compare(int ** orig, int ** new_matrix, int n){
		fw_iterative_serial(orig, n);
		for(int i=1; i <= n; i++){
			for(int j=1; j <= n; j++){
				if(orig[i][j] != new_matrix[i][j]){
					return 0; 
				}
			}
		}
		return 1;
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
	/*
	if(n <= 32){
		printf("Original matrix: \n");
		print_matrix(matrix, n);
	}	
	*/
	
	long long start, end;
	
	//int tilesize[2] = {2, INT_MAX};
	int tilesize[3] = {2, n/S_MATRIX_SIZE, INT_MAX};
	start = clock();
	AFW(dev_matrix, 1, 1, 1, 1, 1, 1, n, 0, tilesize);
	end = clock();
	
	/*
	if(n <= 32){
		int ** new_matrix = copy_matrix_to_host(dev_matrix, n);
		printf("\nWith updated distances: \n");
		print_matrix(new_matrix, n);
		delete[] new_matrix;
	}
	*/

	/*
	if(n <= 1024){
		
		int ** new_matrix = copy_matrix_to_host(dev_matrix, n);
		int ans = compare(matrix, new_matrix, n);
		if(ans)
			printf("ANSWER: CORRECT\n");
		else
			printf("ANSWER: WRONG\n");
		delete[] new_matrix;
	}*/
	cout << "Runtime: " << double(end-start)/double(CLOCKS_PER_SEC) << endl;
	cudaFree(dev_matrix);
	delete[] matrix;
	return 0;
}






