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
		printf("%.3lf\n", vector[i]);

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

__global__ void inverse_matrix_kern(my_float *inverse,my_float *table,my_float *matrix,int size){
	my_float ratio,a;
	int i, j, k, n = size;



	for(i = 0; i < n; i++){
		memcpy(&matrix[i*size*2],&table[i*size],sizeof(my_float)*size);

		for(j = n; j < 2*n; j++){
			if(i==(j-n))
				matrix[i*size*2+j] = 1.0;
			else
				matrix[i*size*2+j] = 0.0;
		}
	}

	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			if(i!=j){
				ratio = matrix[j*size*2+i]/matrix[i*size*2+i];
				for(k = 0; k < 2*n; k++){
					matrix[j*size*2+k] -= ratio * matrix[i*size*2+k];
				}
			}
		}
	}

	for(i = 0; i < n; i++){
		a = matrix[i*size*2+i];
		for(j = 0; j < 2*n; j++){
			matrix[i*size*2+j] /= a;
		}
	}

	for(i = 0; i < n; i++){
		for(j = n; j < 2*n; j++){
			inverse[i*size+j-n] = matrix[i*size*2+j];
		}
	}

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
