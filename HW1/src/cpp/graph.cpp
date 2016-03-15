#include "../headers/graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
//Generate a random matrix of size n
int** generate_matrix(int n){
	int** matrix;
	matrix = new int*[n+1];
	
	int i = 1, j = 1, random;

	//Generate a random matrix
	for(i = 1; i <= n; i++){
		matrix[i] = new int[n+1];
		for(j = 1; j <= n; j++){
			random = rand() % 100;
			matrix[i][j] = random;
		}
	}

	return matrix;
}


//Write matrix to file
void write_matrix_to_file(int ** matrix, int n, char * filename){
	int i,j;
	char * str = NULL;
	ofstream out;
	out.open(filename);
	out << n << endl;
	for(i = 1; i <= n; i++){ 
		for(int j=1; j<= n; j++){
			out << matrix[i][j] << " ";
		}
		out << endl;
	}
	out.close();
}


int** read_matrix_from_file(char * filename){
	int ** matrix = NULL;
	const char * delim = " ";
	int i, j, n;
	string data;
	ifstream in;
	in.open(filename);
	getline(in, data);
	stringstream(data) >> n;
	matrix = new int*[n+1];
	for(i = 1; i<=n; i++){
		matrix[i] = new int[n+1];		
		getline(in, data);
    	stringstream dataStream(data);
    	int val;
    	for(j = 1 ; j <= n; j++){
    		dataStream >> val;
    		matrix[i][j] = val;
    	}
 	}
 	return matrix;
}

int read_matrix_size(char * filename){
	int n;
	string data;
	ifstream in;
	in.open(filename);
	getline(in, data);
	stringstream(data) >> n;
	return n;
}

void print_matrix(int ** matrix, int n){
	int i,j;
	for(i = 1; i <= n; i++){
		for(j = 1; j <= n; j++){
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}	
}


int ** copy_matrix(int ** orig, int n){
	int** matrix;
	matrix = new int*[n+1];
	int i = 1, j = 1;

	//Generate a random matrix
	for(i = 1; i <= n; i++){
		matrix[i] = new int[n+1];
		for(j = 1; j <= n; j++){
			matrix[i][j] = orig[i][j];
		}
	}
	return matrix;
}	

