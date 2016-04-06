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
// extern "C++" void cpp_merge_sort_SM(double * x, int start, int end, int m);

void insertion_sort(double * x, int start, int end){
	int i,j;
	if(start >= end)
		return;
	for(i = start+1; i <= end; i++){
		int currIndex = i;
		for(j = i-1; j >= start; j--){
			if(x[j] > x[currIndex]){
				swap(x, j, currIndex);
				currIndex = j;
			}
		}
	}
}


long binary_search(double key, double *arr, long start, long end)
{
	long mid;
	while(end >= start)
	{
		mid = (start + end)/2;
		if (arr[mid] < key)
		{
			start = mid + 1;	
		}
		else 
		{
            if(arr[mid] == key)
                return mid;
			end = mid - 1;
		}
	}
	return end+1;
}


void print_array(double * x, long start, long end){
	long i;
	for(i=start; i <= end; i++){
		cout << x[i] << " ";
	}
	cout << endl;
}


void scatterKeysToNodes(double * x, long n, int numProcs){
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

void getLocalPivotsFromSlaves(double * allPivots, double *x, long n, long q, int numProcs){
	
	MPI_Status status;
	long size_per_proc = n / (long)numProcs;
	long allPivotPtr = 0;
	long pivotsPerProc = size_per_proc/q;
	long totalNumPivots = n/q;
	long i;
	//put pivots for MASTER in the array.
	for(i = 0; i < size_per_proc; i++){
		if( (i+1)%q == 0)
			allPivots[allPivotPtr++] = x[i];
	}	

	 //Receive local pivots from SLAVES
	for(i = 1; i < numProcs; i++){						
		MPI_Recv((void *)(allPivots + allPivotPtr), pivotsPerProc, MPI_DOUBLE, i, DEFAULT_TAG, MPI_COMM_WORLD, &status);	
		allPivotPtr += pivotsPerProc;
	}

	// cout << "DISPLAYING ALL GATHERED LOCAL PIVOTS: " << endl;
	// print_array(allPivots, 0, totalNumPivots - 1);
}


void distributeGlobalPivotsToSlaves(double * allPivots, int numProcs, long numLocalPivots){
	//Send p globally spaced pivots to all processes
	int numGlobalPivots = numProcs-1;
	long indexIncrement = numLocalPivots / (numProcs-1);
	long beginIndex = indexIncrement-1;
	double * globalPivArray = new double[numGlobalPivots];
	for(int i = 0; i < numGlobalPivots; i++){
		globalPivArray[i] = allPivots[beginIndex];
		beginIndex += indexIncrement;
	}

	//print_array(globalPivArray, 0, numProcs-2);
	for(int i=1; i <=numGlobalPivots ; i++){
		MPI_Send(globalPivArray, numProcs-1, MPI_DOUBLE, i, DEFAULT_TAG, MPI_COMM_WORLD);
	}
}

void sendBucketsToNodes(double * localKeys, int numKeys, double * globalPivots, int numProcs, int myrank){
	long numPivots = numProcs - 1;
	long prevBucketEnd = -1;
	long bucketNumber = -1;
	for (int i = 0; i < numPivots; ++i)
	{
		if(myrank == i) //Bucket of current process remains with itself.
			continue;

		bucketNumber = i;
		double pivot = globalPivots[i];
		long pivotIndex = binary_search(pivot, localKeys, 0, numKeys-1);

		long newEndIndex = -1;
		if(pivotIndex < numKeys){
			if(localKeys[pivotIndex] == pivot)
				newEndIndex = pivotIndex;
			else
				newEndIndex = pivotIndex - 1;
		}	
		else if(pivotIndex == numKeys)
			newEndIndex = pivotIndex-1;		
		else	//SHOULD NEVER REACH HERE
			cout << "Error condition in sendBucketsToNodes()" << endl;
		
		//Send numbers from index: prevBucketEnd+1 to newEndIndex to corresponding bucket.
		long bucketSize = newEndIndex - prevBucketEnd;
		long startIndex = prevBucketEnd + 1; 
		if(bucketSize > 0 && startIndex >= 0 && startIndex < numKeys)
			MPI_Send((void *)(localKeys + startIndex), bucketSize, MPI_DOUBLE, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);

		// if(myrank == 3 && bucketSize > 0 && startIndex >= 0 && startIndex < numKeys){
		// 	cout << "For BUCKET: " << bucketNumber << endl;
		// 	print_array(localKeys, startIndex, startIndex + bucketSize - 1);
		// }
		prevBucketEnd = newEndIndex;
	}
	// cout << "HERE " << prevBucketEnd << endl;
	//Sending the last bucket
	bucketNumber++;
	long lastBucketStart = prevBucketEnd + 1;
	long lastBucketSize = numKeys - lastBucketStart;
	if(lastBucketSize > 0 && lastBucketStart >=0 && lastBucketStart < numKeys)
		MPI_Send((void *)(localKeys + lastBucketStart), lastBucketSize, MPI_DOUBLE, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);

}



int main(int argc, char* argv[]){
	int myrank;
	int numProcs;
	long m = 64;
	long q = 2; // For sending equidistant points after sorting array locally. Select every qth index in local array..
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs); 
		
	if(myrank == MASTER_RANK){
		long n = 16; 
		double * x = generate_array(n);
		print_array(x, 0, n-1);
		
		long size_per_proc = n / (long)numProcs;	

		//Scatter keys to SLAVES.
		scatterKeysToNodes(x, n, numProcs);

		//Sort local keys
		insertion_sort(x, 0, size_per_proc - 1);		
		
		//Create array for storing pivots from all processes including MASTER's pivots
		double * allPivots = new double[n/q];
		getLocalPivotsFromSlaves(allPivots, x, n, q, numProcs);
		
		long totalNumPivots = n/q;
		//Sort pivots
		insertion_sort(allPivots, 0, totalNumPivots-1);
		
		//Distribute global pivots.	
		distributeGlobalPivotsToSlaves(allPivots, numProcs, totalNumPivots);

	}
	else{//**************************************SLAVE PROCESSES************************************************
		long dataSize, i;
		//Get size of data to be received.
		MPI_Recv(&dataSize, 1, MPI_LONG, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		double * localKeys = new double[dataSize];

		//Gather the keys from MASTER
		MPI_Recv(localKeys, dataSize, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
	
		// cpp_merge_sort_SM(localKeys, 0, dataSize-1, m);
		insertion_sort(localKeys, 0, dataSize - 1);		

		//Form an arr of q equidistant points to be distributed to master
		double * arr = new double[dataSize/q];
		long tempIndex = q-1, counter = 0;
		while(tempIndex < dataSize){
			arr[counter++] = localKeys[tempIndex];
			tempIndex += q;
		}

		// if(myrank == 1){
		// 	cout << "Equidistant points arr: " << endl;
		// 	for(i=0; i < dataSize/q; i++)
		// 		cout << arr[i] << " ";
		// 	cout << endl;
		// }

		//Send equidistant pivots to MASTER
		MPI_Send((void *)arr, dataSize/q, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD);
		
		//Receive global pivots from MASTER
		double * globalPivArray = new double[numProcs - 1];
		MPI_Recv(globalPivArray, numProcs-1, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		
		//Distribute buckets to appropriate nodes
		sendBucketsToNodes(localKeys, dataSize, globalPivArray, numProcs, myrank);

	}
	
	//cout << "Rank: " << myrank << endl;
	MPI_Finalize();
	return 0;
}


