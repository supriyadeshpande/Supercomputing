#include <iostream>
#include <fstream>
#include <mpi.h>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <string.h>
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


long binary_search(double key, double *arr, long start, long end){
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
	long numLocalPivots = n/q;
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
	// print_array(allPivots, 0, numLocalPivots - 1);
}

void initializeGlobalPivots(double * allPivots, long numLocalPivots, double * globalPivArray, int numGlobalPivots){
	long indexIncrement = numLocalPivots / numGlobalPivots;
	long beginIndex = indexIncrement-1;
	for(int i = 0; i < numGlobalPivots; i++){
		globalPivArray[i] = allPivots[beginIndex];
		beginIndex += indexIncrement;
	}
}


double distributeGlobalPivotsToSlaves(double * globalPivArray, int numGlobalPivots, int numProcs){
	//Send p globally spaced pivots to all processes

	//print_array(globalPivArray, 0, numProcs-2);
	for(int i=1; i < numProcs ; i++){
		MPI_Send(globalPivArray, numGlobalPivots, MPI_DOUBLE, i, DEFAULT_TAG, MPI_COMM_WORLD);
	}
}

void sendBucketsToNodes(double * localKeys, long numKeys, double * globalPivots, int numProcs, int myrank, double * myBucket, long * myBucketPtr){
	long numPivots = numProcs - 1;
	long prevBucketEnd = -1;
	long bucketNumber = -1;
	for (int i = 0; i < numPivots; ++i)
	{
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
		if(bucketSize > 0 && startIndex >= 0 && startIndex < numKeys){
			if(bucketNumber != myrank){
				MPI_Send((void *)&bucketSize, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);
				MPI_Send((void *)(localKeys + startIndex), bucketSize, MPI_DOUBLE, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);
			}
			else{
				memcpy((void *)myBucket, (void *)(localKeys + startIndex), bucketSize*sizeof(double));
				*myBucketPtr += bucketSize;
			}
		}
		else{
			if(bucketNumber != myrank){
				long negative_size = -1;
				MPI_Send((void *)&negative_size, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);
			}
		}

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
	if(lastBucketSize > 0 && lastBucketStart >=0 && lastBucketStart < numKeys){
		if(bucketNumber != myrank){
			MPI_Send((void *)&lastBucketSize, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);
			MPI_Send((void *)(localKeys + lastBucketStart), lastBucketSize, MPI_DOUBLE, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);
		}
		else{
			memcpy((void *)myBucket, (void *)(localKeys + lastBucketStart), lastBucketSize*sizeof(double));
			*myBucketPtr += lastBucketSize;
		}
	}
	else{
		if(bucketNumber != myrank){
			long negative_size = -1;
			MPI_Send((void *)&negative_size, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD);
		}
	}
}

//Receive buckets for this process, bucketPtr points to start index where new values starthould be copied.
void receiveMyBuckets(double * myBucket, long bucketPtr, int numProcs, int myrank){
	MPI_Status status;
	long size = -1;
	for(int i = 0; i < numProcs; i++){
		if(i == myrank)
			continue;
		MPI_Recv(&size, 1, MPI_LONG, i, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		if(size > 0){
			// if(myrank == 0){
			// 	cout << "size received: " << size << " From process: " << i << endl;
			// }
			MPI_Recv((void *)(myBucket + bucketPtr), size, MPI_DOUBLE, i, DEFAULT_TAG, MPI_COMM_WORLD, &status);	
			bucketPtr += size;
		}
	}
}

void sendSortedBucketToMaster(double * myBucket, long bucketSize){

	if(bucketSize > 0){
		MPI_Send((void *)&bucketSize, 1, MPI_LONG, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD);	
		MPI_Send((void *)myBucket, bucketSize, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD);
	}
	else{
		long negative_size = -1;
		MPI_Send((void *)&negative_size, 1, MPI_LONG, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD);
	}
}

void createSortedArr(double * x, long n, double * myBucket, long myBucketSize, int numProcs){
	MPI_Status status;
	long i;
	int j;
	for(i = 0; i < myBucketSize; i++){
		x[i] = myBucket[i];
	}

	for(j=1; j < numProcs; j++){
		long size=0;
		MPI_Recv((void *)&size, 1, MPI_LONG, j, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		if(size > 0){
			MPI_Recv((void *)(x + i), size, MPI_DOUBLE, j, DEFAULT_TAG, MPI_COMM_WORLD, &status);
			i += size;
		}
	}
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
		long n = 128; 
		double * x = generate_array(n);

		cout << "Input array: " << endl;
		print_array(x, 0, n-1);
		
		long size_per_proc = n / (long)numProcs;	

		//Send value of n to last node
		MPI_Send(&n, 1, MPI_LONG, numProcs-1, DEFAULT_TAG, MPI_COMM_WORLD);

		//Scatter keys to SLAVES.
		scatterKeysToNodes(x, n, numProcs);

		//Sort local keys
		insertion_sort(x, 0, size_per_proc - 1);		
		
		//Create array for storing pivots from all processes including MASTER's pivots
		double * allPivots = new double[n/q];
		getLocalPivotsFromSlaves(allPivots, x, n, q, numProcs);
		
		long numLocalPivots = n/q;
		//Sort pivots
		insertion_sort(allPivots, 0, numLocalPivots-1);
		
		long numGlobalPivots = numProcs-1;
		double * globalPivots = new double[numGlobalPivots];
		initializeGlobalPivots(allPivots, numLocalPivots, globalPivots, numGlobalPivots);

		//Distribute global pivots.	
		distributeGlobalPivotsToSlaves(globalPivots, numGlobalPivots, numProcs);

		
		long myBucketSize = globalPivots[myrank];
		double * myBucket = new double[myBucketSize];
		long bucketPtr = 0;

		//Send buckets to appropriate nodes.
		long numKeys = size_per_proc; // First size_per_proc keys in array x belong to master.
		sendBucketsToNodes(x, numKeys, globalPivots, numProcs, myrank, myBucket, &bucketPtr);//CORRECT THIS.

		//Receive my buckets from other processes
		receiveMyBuckets(myBucket, bucketPtr, numProcs, myrank);


		//Sort my bucket
		insertion_sort(myBucket, 0, myBucketSize-1); //NOTE: Can be optimized by putting in createSortedArr function.

		//Receive final sorted buckets and create final array
		createSortedArr(x, n, myBucket, myBucketSize, numProcs);

		cout << "Sorted array: " << endl;
		print_array(x, 0 , n-1);
	}
	else{//**************************************SLAVE PROCESSES************************************************
		long n, dataSize, i;

		//Get value of n
		if(myrank == numProcs-1)
			MPI_Recv(&n, 1, MPI_LONG, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		
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

		//Send equidistant pivots to MASTER
		MPI_Send((void *)arr, dataSize/q, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD);
		
		//Receive global pivots from MASTER
		double * globalPivArray = new double[numProcs - 1];
		MPI_Recv(globalPivArray, numProcs-1, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		

		long myBucketSize = myrank != (numProcs-1) ? globalPivArray[myrank] - globalPivArray[myrank-1] : n - globalPivArray[myrank-1];

		double * myBucket = new double[myBucketSize];
		long bucketPtr = 0;

		//Distribute buckets to appropriate nodes
		sendBucketsToNodes(localKeys, dataSize, globalPivArray, numProcs, myrank, myBucket, &bucketPtr);

		//Receive my buckets from other processes
		receiveMyBuckets(myBucket, bucketPtr, numProcs, myrank);

		//Sort my bucket locally
		insertion_sort(myBucket, 0, myBucketSize-1);

		//Send my bucket to MASTER
		sendSortedBucketToMaster(myBucket, myBucketSize);
	}
	MPI_Finalize();
	return 0;
}


