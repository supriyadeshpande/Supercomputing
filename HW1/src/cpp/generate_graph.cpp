#include "../headers/graph.h"
#include <iostream>

using namespace std;
int main(){
	int n;
	char prefix[] = "../input/";
	char filename[100];
	int ** matrix = NULL;

	cout << "Enter number of graph nodes: " << endl;
	cin >> n;
	cout << "Enter filename: " << endl;
	cin >> filename;
	strcat(prefix, filename);
	matrix = generate_matrix(n);
	//print_matrix(matrix, n);
	write_matrix_to_file(matrix, n, prefix);
	return 0;
}
