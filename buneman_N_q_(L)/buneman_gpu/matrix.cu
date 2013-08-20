#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CULA_USE_CUDA_COMPLEX 

#include <cula.h>
#include <cula_lapack.h>
#include "memory.h"
#include "matrix.h"


cublasHandle_t handle;

int pow(int n,int p){
	return (int) pow((my_float)n,(my_float)p);
}

my_float log2(int x){
	return  log((my_float)x) / (my_float) log(2.0);
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

int scalarMatrix_multiplication(my_float *multiply,my_float *tableA,int rowsA,int colsA,my_float scalar){
	my_float *tableB;
	my_float beta = 0.0;
	cudaMalloc((void **) &tableB,rowsA*colsA*sizeof(my_float));
	cudaMemset(tableB,0,rowsA*colsA*sizeof(my_float));
	cublasStatus_t ret;

#ifdef sprecision
	ret = cublasSgeam(handle, CUBLAS_OP_N , CUBLAS_OP_N ,rowsA,colsA,&scalar, tableA,rowsA,&beta,tableB,rowsA,multiply,rowsA);          
#else
	ret = cublasDgeam(handle, CUBLAS_OP_N , CUBLAS_OP_N ,rowsA,colsA,&scalar, tableA,rowsA,&beta,tableB,rowsA,multiply,rowsA);          

#endif

	if (ret != CUBLAS_STATUS_SUCCESS){
		printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
		return 1;
	}

	cudaFree(tableB);
	return 1;
}

int matrixVector_multiplication(my_float *multiply,my_float *tableA,my_float *vectorB,int rowsA,int colsA,int rowsB, my_float alpha){
	my_float beta;
	beta = 0.0;

	cublasStatus_t ret;
#ifdef sprecision
	ret = cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N , rowsA , 1 , rowsA, &alpha , tableA, rowsA, vectorB, rowsA , &beta , multiply , rowsA);
#else
	ret = cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N , rowsA , 1 , rowsA, &alpha , tableA, rowsA, vectorB, rowsA , &beta , multiply , rowsA);
#endif

	if (ret != CUBLAS_STATUS_SUCCESS){
		printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
		return 1;
	}

	return 1;
}



int inverse_matrix(my_float *inverse,int *ipiv,int size){
	culaStatus s;	

#ifdef sprecision
	s = culaDeviceSgetrf(size,size,inverse,size,ipiv);
	if( s != culaNoError ){
		if( s == culaDataError )
			printf("Data error with code %d, please see LAPACK documentation\n",culaGetErrorInfo());
		else
			printf("%s\n", culaGetStatusString(s));
	}

	s = culaDeviceSgetri(size,inverse,size,ipiv);
	if( s != culaNoError ){
		if( s == culaDataError )
			printf("Data error with code %d, please see LAPACK documentation\n",culaGetErrorInfo());
		else
			printf("%s\n", culaGetStatusString(s));
	}
#else
	s = culaDeviceDgetrf(size,size,inverse,size,ipiv);
	if( s != culaNoError ){
		if( s == culaDataError )
			printf("Data error with code %d, please see LAPACK documentation\n",culaGetErrorInfo());
		else
			printf("%s\n", culaGetStatusString(s));
	}

	s = culaDeviceDgetri(size,inverse,size,ipiv);
	if( s != culaNoError ){
		if( s == culaDataError )
			printf("Data error with code %d, please see LAPACK documentation\n",culaGetErrorInfo());
		else
			printf("%s\n", culaGetStatusString(s));
	}
#endif
	return 1;
}

int matrix_multiplication(my_float *multiply,my_float *tableA,my_float *tableB,int rowsA,int colsA,int rowsB,int colsB){
	my_float alpha,beta;
	alpha = 1.0;
	beta = 0.0;

	cublasStatus_t ret;

#ifdef sprecision
	ret = cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N , rowsA , rowsA , rowsA, &alpha , tableA, rowsA,tableB, rowsA , &beta , multiply , rowsA);
#else
	ret = cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N , rowsA , rowsA , rowsA, &alpha , tableA, rowsA,tableB, rowsA , &beta , multiply , rowsA);
#endif

	if (ret != CUBLAS_STATUS_SUCCESS){
		printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
		return 1;
	}
	return 1;
}

int matrix_addition(my_float *add,my_float *tableA,my_float *tableB,int rowsA,int colsA,int rowsB,int colsB,my_float alpha,my_float beta){

	cublasStatus_t ret;

#ifdef sprecision
	ret = cublasSgeam(handle, CUBLAS_OP_N , CUBLAS_OP_N ,rowsA,colsA,&alpha, tableA,rowsA,&beta,tableB,rowsB,add,rowsA);          
#else
	ret = cublasDgeam(handle, CUBLAS_OP_N , CUBLAS_OP_N ,rowsA,colsA,&alpha, tableA,rowsA,&beta,tableB,rowsB,add,rowsA);          
#endif

	if (ret != CUBLAS_STATUS_SUCCESS){
		printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
		return 1;
	}
	return 1;

}

int vector_addition(my_float *add,my_float *vectorA,my_float *vectorB,int sizeA,int sizeB,my_float alpha,my_float beta){

	cublasStatus_t ret;
#ifdef sprecision
	ret = cublasSgeam(handle, CUBLAS_OP_N , CUBLAS_OP_N ,sizeA,1,&alpha, vectorA,sizeA,&beta,vectorB,sizeB,add,sizeB);
#else
	ret = cublasDgeam(handle, CUBLAS_OP_N , CUBLAS_OP_N ,sizeA,1,&alpha, vectorA,sizeA,&beta,vectorB,sizeB,add,sizeB);
#endif

	if (ret != CUBLAS_STATUS_SUCCESS){
		printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
		return 1;
	}
	return 1;
}
