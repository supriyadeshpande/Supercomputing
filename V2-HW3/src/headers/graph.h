#ifndef GRAPH_H    
#define GRAPH_H

//Reads an adjacency matrix from file
int** read_matrix_from_file(char * filename);

//Read size of adj matrix from file : line 1
int read_matrix_size(char * filename);

//Writes an adjacency matrix to file named filename
void write_matrix_to_file(int ** matrix, int n, char * filename);

//Generates a random n*n matrix
int** generate_matrix(int n);	

//Prints adjacency for n*n matrix
void print_matrix(int ** matrix, int n);

//Copies an adjacency matrix into another.
int ** copy_matrix(int ** orig, int n);

#endif
