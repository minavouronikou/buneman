#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <mkl.h>
#include <mkl_scalapack.h>
//#include "memory.h"
#include "matrix.h"

int pow(int n,int p){
	return (int) pow((my_float)n,(my_float)p);
}

my_float log2(int x){
	return  log((my_float)x) / log(2.0);
}

void print_matrix(my_float *table,int rows,int cols){
	int i,j;
	printf("Matrix :\n");

	for ( i = 0 ; i < rows ; i++ ){
		for ( j = 0 ; j < cols ; j++ ){
			printf("%.3lf\t", table[i*rows+j]);
		}
		printf("\n");
	}

}

void print_vector(my_float *vector,int size){
	int i;
	printf("Vector :\n");

	for (i = 0 ; i < size ; i++ )
		printf("%.3lf\t", vector[i]);

	printf("\n");
}




#ifdef MKL
/**************************************************** MKL Matrices ********************************************************************/
int matrixVector_multiplication(my_float *multiply,my_float *tableA,my_float *vectorB,int rowsA,int colsA,int rowsB,my_float alpha){
	my_float beta;
	beta = 0.0;

#ifdef sprecision
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,rowsA,	  1, rowsA, alpha, tableA, rowsA, vectorB, rowsA,beta,multiply,rowsA);
#else
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,rowsA,	  1, rowsA, alpha, tableA, rowsA, vectorB, rowsA,beta,multiply,rowsA);
#endif

	return 1;
}


int inverse_matrix(my_float *inverse,int *p,my_float *work,int lwork,int size){
	int status;
	
#ifdef sprecision
	sgetrf(&size,&size,inverse,&size,p,&status);
	sgetri(&size,inverse,&size,p,work,&lwork,&status);


#else
	dgetrf(&size,&size,inverse,&size,p,&status);
	dgetri(&size,inverse,&size,p,work,&lwork,&status);
#endif

	return 1;
}

int matrix_multiplication(my_float *multiply,my_float *tableA,my_float *tableB,int rowsA,int colsA,int rowsB,int colsB){
	my_float alpha,beta;
	alpha = 1.0;
	beta = 0.0;

	#ifdef sprecision
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,rowsA, rowsA, rowsA, alpha, tableA, rowsA, tableB, rowsA, beta, multiply,rowsA);
	#else
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,rowsA, rowsA, rowsA, alpha, tableA, rowsA, tableB, rowsA, beta, multiply,rowsA);
	#endif

	return 1;
}
int matrix_addition(my_float *add,my_float *tableA,my_float *tableB,int rowsA,int colsA,int rowsB,int colsB,my_float alpha,my_float beta){
	int i,j;

	#ifdef sprecision
		mkl_somatadd('R','N', 'N', rowsA, colsA, alpha, tableA, rowsA, beta, tableB, rowsA, add, rowsA);
	#else
		mkl_domatadd('R','N', 'N', rowsA, colsA, alpha, tableA, rowsA, beta,tableB, rowsA, add, rowsA);
	#endif

	return 1;

}
int vector_addition(my_float *add,my_float *vectorA,my_float *vectorB,int sizeA,int sizeB,int alpha,int beta){
	

	#ifdef sprecision
		mkl_somatadd('C','N', 'N', sizeA, 1, alpha,vectorA, sizeA, beta, vectorB, sizeB, add, sizeA);
	#else
		mkl_domatadd('C','N', 'N', sizeA, 1, alpha,vectorA, sizeA, beta, vectorB, sizeB, add, sizeA);
	#endif
				

	return 1;
}

int scalarMatrix_multiplication(my_float *multiply,my_float *tableA,int rowsA,int colsA,my_float scalar){
	my_float *tableB;
	my_float beta = 0.0;
	//tableB = (my_float*)malloc(rowsA*colsA*sizeof(my_float));
	tableB= (my_float*)mkl_malloc( sizeof(my_float)*(rowsA*colsA), 128);
	memset(tableB,0,rowsA*colsA*sizeof(my_float));
	

#ifdef sprecision
	cblas_saxpby(rowsA*colsA,scalar,tableA,1,0.0,multiply,1);
#else
	cblas_daxpby(rowsA*colsA,scalar,tableA,1,0.0,multiply,1);
#endif
	mkl_free(tableB);
	return 1;
}

#endif