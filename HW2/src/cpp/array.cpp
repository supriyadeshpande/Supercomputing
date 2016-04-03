#include "../headers/array.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>

using namespace std;
//Generate a random matrix of size n

void swap(double * x, long p1, long p2){
	double temp = x[p1];
	x[p1] = x[p2];
	x[p2] = temp;
}

double * generate_array(int n){
	double * arr = new double[n];	
	int i = 1, random;

	//Generate array such that arr[i] = i+1
	for(i = 0; i < n; i++)
		arr[i] = (double)(i+1);

	//Randomly take any two positions and swap
	for(i = 0; i < n; i++){
		int p1 = rand() % n;
		int p2 = rand() % n;
		if(p1 != p2)
			swap(arr, p1, p2);
	}
	return arr;
}

void print_array(double* x, int n){
	int i;
	for(i = 0; i < n; i++)
		cout << x[i] << " ";
	cout << endl;
}
