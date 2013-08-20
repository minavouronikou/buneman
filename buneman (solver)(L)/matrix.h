#include "global.h"

my_float log2(int x);
int pow(int n,int p);
void print_matrix(my_float *table,int rows,int cols);
void print_vector(my_float *vector,int size);

/**************************************************** MKL Matrices ********************************************************************/
int scalarMatrix_multiplication(my_float *multiply,my_float *tableA,int rowsA,int colsA,my_float alpha);
int matrixVector_multiplication(my_float *output,my_float *tableA,my_float *vectorB,int rowsA,int colsA,int rowsB,my_float alpha);
int matrix_multiplication(my_float *output,my_float *tableA,my_float *tableB,int rowsA,int colsA,int rowsB,int colsB);
int inverse_matrix(my_float *inverse,int *p,my_float *work,int lwork,int size);
int matrix_addition(my_float *add,my_float *tableA,my_float *tableB,int rowsA,int colsA,int rowsB,int colsB,my_float alpha,my_float beta);

int vector_addition(my_float *output,my_float *vectorA,my_float *vectorB,int sizeA,int sizeB,int alpha,int beta);


int calcB(my_float *add,my_float * B,my_float * T,int size,int index);
int solver_system(my_float *vectorX,my_float *tableA,int rowsA,int colsA,int rowsX,int *p);
/****************************************************************************************************************************/



