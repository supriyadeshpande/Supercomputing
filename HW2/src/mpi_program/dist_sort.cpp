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

//************************************************************************************************************************



long binary_search_par(double key, double *arr, long start, long end){
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

//************************************************************************************************************************



void merge(double * t, long p1, long r1, long p2, long r2, double * x, long p3){
        // cout << "Input to merge: \n";
        long i=p1, j=p2, k=p3;
        while(i <= r1 && j <= r2){
            if(t[i] <= t[j]) {
                x[k++] = t[i++];
            }
            else {
                x[k++] = t[j++];
            }
        }

        while(i <= r1)  {
            x[k++] = t[i++];
        }

        while(j <= r2) {
            x[k++] = t[j++];
        }
}

void par_merge(double * t, long p1, long r1, long p2, long r2, double *x, long p3, long m2)
{
        // cout << "Par merge input: \n";
       
    /*
        if(r2 < p2 && r1 < p1)
            return;
        else if(r2 < p2){
            cout << "Here1\n";
            memcpy((void *)(x + p3), (void *)(t + p1), (r1-p1+1)*sizeof(double));
            return;
        }
        else if(r1 < p1){
            cout << "Here2\n";
            memcpy((void *)(x + p3), (void *)(t + p2), (r2-p2+1)*sizeof(double));
            return;
        }
*/

        long n1 = r1 - p1 + 1;
        long n2 = r2 - p2 + 1;

        // if(n1 ==3 && n2 ==2){
        //     cout << "ss1\n";
        //     print_array(t, p1, r1);
        //      cout << "ss2\n";
        //     print_array(t, p2, r2);
        // }
        


        long r3 = p3 + n1 + n2 - 1;
        long q1, q2, q3;
        long p3_copy = p3;

        if (n1 + n2 <= m2) {
                merge(t, p1, r1, p2, r2, x, p3);
                // cout << "Normal merge output: " << endl;
                // print_array(x, p3, r3);
        }
        else {
            if (n1 < n2)
            {
                swap(&p1, &p2);
                swap(&r1, &r2);
                swap(&n1, &n2);
            }

            if(n1 == 0){
                // cout << "Returning here\n";
                return;
            }

            q1 = floor(( r1 + p1 ) / 2) ;
            q2 = binary_search_par(t[q1], t, p2, r2);
            q3 = p3 + (q1 - p1) + (q2 - p2);

            x[q3] = t[q1];

            // if(t[p1] == 1 && t[p2] == 4){
            //     cout << "q1: " << q1 << " q2: " << q2 << "q3: " <<q3 << endl;   
            // }

            par_merge(t, p1, q1-1, p2, q2-1, x, p3, m2);
            par_merge(t, q1 + 1, r1, q2, r2, x, q3+1, m2);
        }
        
        // cout << "Par merge output: \n";
        // cout << "p3: " <<p3<< " r3: "<<r3 << endl;
        // print_array(x, p3, r3);

}

void par_merge_sort_PM(double * x, long start, long end, long m2, long m3){

        int i;
        long n = end - start + 1;
        double * t = new double[n];

        if(n <= m3){
                insertion_sort(x, start, end);
                return;
        }

        if(n == 1)
                return;

        long mid = start + (end - start)/2;

        par_merge_sort_PM(x, start, mid, m2, m3);
        par_merge_sort_PM(x, mid+1, end, m2, m3);

        long leftSize = mid - start + 1;

        // cout << "p1: ";
        // print_array(x, start, mid);
        // cout << "p2: ";
        // print_array(x, mid+1, end);
        // cout << "\n" << n << endl;

        for(i = 0; i < n; i++) {
                t[i] = x[start + i];
        }

        mid = mid - start + 1;
        // cout << "Before merge\n";
        // print_array(x, start, mid);
        // print_array(x, mid+1, end);

        par_merge(t, 0, mid-1, mid, n-1, x, start, m2);
        
        // cout << "After merge\n";
        // print_array(x, start, mid);

        delete[] t;
}

//************************************************************************************************************************



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
	MPI_Request req;
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
                MPI_Isend((void *)&bucketSize, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD, &req);
                MPI_Isend((void *)(localKeys + startIndex), bucketSize, MPI_DOUBLE, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD, &req);

			}
			else{
				memcpy((void *)myBucket, (void *)(localKeys + startIndex), bucketSize*sizeof(double));
				*myBucketPtr += bucketSize;
			}
		}
		else{
			if(bucketNumber != myrank){
				long negative_size = -1;
				MPI_Isend((void *)&negative_size, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD, &req);
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
			MPI_Isend((void *)&lastBucketSize, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD, &req);
			MPI_Isend((void *)(localKeys + lastBucketStart), lastBucketSize, MPI_DOUBLE, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD, &req);
		}
		else{
			memcpy((void *)myBucket, (void *)(localKeys + lastBucketStart), lastBucketSize*sizeof(double));
			*myBucketPtr += lastBucketSize;
		}
	}
	else{
		if(bucketNumber != myrank){
			long negative_size = -1;
			MPI_Isend((void *)&negative_size, 1, MPI_LONG, bucketNumber, DEFAULT_TAG, MPI_COMM_WORLD, &req);
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
			
			// if(!myrank){
			// 	cout << "Received bucket of size: " << size << endl;
			// 	print_array(myBucket, bucketPtr, bucketPtr + size - 1);
			// }
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
	long m2 = 16;
	long m3 = 8;
	long q = 2; // For sending equidistant points after sorting array locally. Select every qth index in local array..
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs); 
		
	if(myrank == MASTER_RANK){
		
		long n = atoi(argv[1]);
		cout << "N: " << (n+1);
		/*
		long n = 32; 
		double * x = generate_array(n);
		long long start_t, end_t;
		cout << "Input array: " << endl;
		print_array(x, 0, n-1);
		
		long size_per_proc = n / (long)numProcs;	

		start_t = clock();
		//Send value of n to last node
		MPI_Send(&n, 1, MPI_LONG, numProcs-1, DEFAULT_TAG, MPI_COMM_WORLD);

		//Scatter keys to SLAVES.
		scatterKeysToNodes(x, n, numProcs);

		//Sort local keys		
		// insertion_sort(x, 0, size_per_proc - 1);		
		par_merge_sort_PM(x, 0, size_per_proc-1, m2, m3);

		//Create array for storing pivots from all processes including MASTER's pivots
		double * allPivots = new double[n/q];
		getLocalPivotsFromSlaves(allPivots, x, n, q, numProcs);
		
		long numLocalPivots = n/q;
		//Sort pivots
		// insertion_sort(allPivots, 0, numLocalPivots-1);
		par_merge_sort_PM(allPivots, 0, numLocalPivots-1, m2, m3);
		
		long numGlobalPivots = numProcs-1;
		double * globalPivots = new double[numGlobalPivots];
		initializeGlobalPivots(allPivots, numLocalPivots, globalPivots, numGlobalPivots);

		delete[] allPivots;
		
		//Distribute global pivots.	
		distributeGlobalPivotsToSlaves(globalPivots, numGlobalPivots, numProcs);

		
		long myBucketSize = globalPivots[myrank];
		// cout << "Bucket size: " << myBucketSize << endl;
		double * myBucket = new double[myBucketSize];
		long bucketPtr = 0;
		// cout << "Check 1: \n"; 
		//Send buckets to appropriate nodes.
		long numKeys = size_per_proc; // First size_per_proc keys in array x belong to master.
		sendBucketsToNodes(x, numKeys, globalPivots, numProcs, myrank, myBucket, &bucketPtr);//CORRECT THIS.

		delete[] globalPivots;

		//Receive my buckets from other processes
		receiveMyBuckets(myBucket, bucketPtr, numProcs, myrank);


		//Sort my bucket
		// cout << "Before local bucket sorting: \n";
		// print_array(myBucket, 0, myBucketSize-1);
		// insertion_sort(myBucket, 0, myBucketSize-1); //NOTE: Can be optimized by putting in createSortedArr function.
		par_merge_sort_PM(myBucket, 0, myBucketSize-1, m2, m3);

		// cout << "After local bucket sorting: \n";
		// print_array(myBucket, 0, myBucketSize-1);

		//Receive final sorted buckets and create final array
		createSortedArr(x, n, myBucket, myBucketSize, numProcs);

		end_t = clock();
		// cout << "INPUT SIZE: " << n << " TIME: " << double(end_t - start_t) / double(CLOCKS_PER_SEC);
		cout << "Sorted array: " << endl;
		print_array(x, 0 , n-1);

		delete[] myBucket;
		delete[] x;
		*/
	}
	else{//**************************************SLAVE PROCESSES************************************************
		/*
		long n, dataSize, i;

		//Get value of n
		if(myrank == numProcs-1)
			MPI_Recv(&n, 1, MPI_LONG, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		
		//Get size of data to be received.
		MPI_Recv(&dataSize, 1, MPI_LONG, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		double * localKeys = new double[dataSize];

		//Gather the keys from MASTER
		MPI_Recv(localKeys, dataSize, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
	
		// insertion_sort(localKeys, 0, dataSize - 1);	
		par_merge_sort_PM(localKeys, 0, dataSize - 1, m2, m3);	

		//Form an arr of q equidistant points to be distributed to master
		double * arr = new double[dataSize/q];
		long tempIndex = q-1, counter = 0;
		while(tempIndex < dataSize){
			arr[counter++] = localKeys[tempIndex];
			tempIndex += q;
		}

		//Send equidistant pivots to MASTER
		MPI_Send((void *)arr, dataSize/q, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD);
		
		delete[] arr;

		//Receive global pivots from MASTER
		double * globalPivArray = new double[numProcs - 1];
		MPI_Recv(globalPivArray, numProcs-1, MPI_DOUBLE, MASTER_RANK, DEFAULT_TAG, MPI_COMM_WORLD, &status);
		

		long myBucketSize = myrank != (numProcs-1) ? globalPivArray[myrank] - globalPivArray[myrank-1] : n - globalPivArray[myrank-1];

		double * myBucket = new double[myBucketSize];
		long bucketPtr = 0;

		//Distribute buckets to appropriate nodes
		sendBucketsToNodes(localKeys, dataSize, globalPivArray, numProcs, myrank, myBucket, &bucketPtr);

		delete[] globalPivArray;
		//Receive my buckets from other processes
		receiveMyBuckets(myBucket, bucketPtr, numProcs, myrank);

		//Sort my bucket locally
		// insertion_sort(myBucket, 0, myBucketSize-1);
		par_merge_sort_PM(myBucket, 0, myBucketSize-1, m2, m3);

		//Send my bucket to MASTER
		sendSortedBucketToMaster(myBucket, myBucketSize);
		
		delete[] myBucket;
		delete[] localKeys;
		*/
	}
	MPI_Finalize();
	return 0;
}


