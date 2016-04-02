#include <iostream>
#include <fstream>
#include <mpi.h>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include "../headers/array.h"

#define MASTER_RANK 0
#define DEFAULT_TAG 0

using namespace std;

long min(long a, long b){return a <= b ? a : b;}

int main(int argc, char* argv[]){
	int myrank;
	int numProcs;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs); 
		
	if(myrank == MASTER_RANK){
		long n = atoi(argv[1]); 
		double * x = generate_array(n);
		long size_per_proc = n / (long)numProcs;	
		long i = size_per_proc,j, processRank=1;
		//Scatter the keys
		
		while(i < n){
			long start = i;
			long end = min(n-1, start + size_per_proc - 1);
			long currentSize = end - start + 1; 
			int processRank = start / size_per_proc;//Rank of process to send data to.
			//First send size of data
			MPI_Send(&currentSize, 1, MPI_LONG, processRank, DEFAULT_TAG, MPI_COMM_WORLD);
			//Send actual data.
			MPI_Send((void *)(x+start), currentSize, MPI_DOUBLE, processRank, DEFAULT_TAG, MPI_COMM_WORLD);	
			i += size_per_proc;	
		}
	}
	else{
		long dataSize, i;
		//Get size of data to be received.
		MPI_Recv(&dataSize, 1, MPI_LONG, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		double * localKeys = new double[dataSize];
		//Gather the keys
		MPI_Recv(localKeys, dataSize, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		//cout << "Received array: " << endl;
		//for(i = 0; i < dataSize; i++)
		//	cout << localKeys[i] << " ";
		cout << endl;	
	}
	
	//cout << "Rank: " << myrank << endl;
	MPI_Finalize();
	return 0;
}
