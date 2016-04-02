
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

void swap(double * x, long p1, long p2){
	double temp = x[p1];
	x[p1] = x[p2];
	x[p2] = temp;
}

void insertion_sort(double * x, long start, long end){
        long i,j;
        for(i = start+1; i <= end; i++){
                long currIndex = i;
                for(j = i-1; j >= 0; j--){
                        if(x[j] > x[currIndex]){
                                swap(x, j, currIndex);
                                currIndex = j;
                        }
                        else
                                break;
                }
        }
}

void print(double * x, int n){
	int i=0;
	for(i=0; i < n; i++)
		cout << x[i] << " ";
	cout << endl;
}

int main(){
	double * x = new double[10];
	int i;
	for(i=0; i < 10; i++)
		x[i] = double(10-i);
	print(x,10);
	insertion_sort(x, 0, 9);
	print(x,10);
}