#include <iostream>
#include <fstream>

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
void write_matrix(int ** matrix, int n, char * filename){
	int i,j;
	char * str = NULL;
	std::ofstream out;
	out.open(filename);
	out << n << std::endl;
	for(i = 1; i <= n; i++){ 
		for(int j=1; j<= n; j++){
			out << matrix[i][j] << " ";
		}
		out << std::endl;
	}
	out.close();
}

int main(){
	int n;
	int ** matrix = NULL;
	char * filename = (char *)malloc(50);
	strcpy(filename, "input1.txt");
	std::cout << "Enter n:" << std::endl;
	std::cin >> n;
	matrix = generate_matrix(n);
	write_matrix(matrix, n, filename);
	return 0; 
}
