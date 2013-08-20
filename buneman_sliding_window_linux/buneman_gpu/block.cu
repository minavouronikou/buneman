/*
*	Buneman's algorithm for block tridiagonal systems 
*	This is a variation of block cyclic reduction that improves stability
*	11/06/2013
*	 MKL library is used for matrix operations
*	
*
*/

#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_runtime.h>

#define CULA_USE_CUDA_COMPLEX 
#include <cula.h>

#include "matrix.h"
#include "memory.h"

#define PI 3.14159265358979323846

extern cublasHandle_t handle;


//this function is used to convert 3D representation to 1D
__inline int i3to1(int i,int j,int k,int n,int q){
	return i*n*q+j*q+k;
}

my_float calc_theta(int k,int l){
	my_float theta;
	theta = (l-0.5)*PI/(my_float) pow(2,k);
	return theta;
}

my_float calc_alpha(int k,int l,my_float theta){
	my_float alpha;
	alpha = sin(theta)*pow(-1,l)/pow(2,k);	
	return alpha;

}

int calcT(my_float *Tres,my_float *T1,my_float *T2,int size,int index){
	int p;

	cudaMemcpy(T1,T2,sizeof(my_float)*size*size,cudaMemcpyDeviceToDevice);
	cudaMemcpy(Tres,T2,sizeof(my_float)*size*size,cudaMemcpyDeviceToDevice);
	int exponent = pow(2,index);
	for(p=0;p<exponent-1;p++){

		matrix_multiplication(Tres,T1,T2,size,size,size,size);
		cudaMemcpy(T1,Tres,sizeof(my_float)*size*size,cudaMemcpyDeviceToDevice);
	}
	return 1;
}

int calcInvB(my_float *add,my_float *B,my_float *T,int size,int index,int multiply,int *ipiv,my_float *temp){
	int l;
	my_float theta,cosine,alpha;
	int to = pow(2,index);

	cudaMemset(add,0,size*size*sizeof(my_float));

	for(l=1;l<=to;l++){
		theta = calc_theta(index,l);
		cosine = -2*cos(theta);
		alpha = calc_alpha(index,l,theta);

		matrix_addition(temp,B,T,size,size,size,size,1,cosine*multiply);
		inverse_matrix(temp,ipiv,size);
		matrix_addition(add,add,temp,size,size,size,size,1,-alpha);
	}

	return 1;
}


void check_mem(void *p,const char *error){
	if(p==NULL){
		printf("Memory malloc error %s\n",error);
		exit(1);
	}
}

__global__ void print_kernel(my_float *vector,int size){
	int i;
	for(i=0;i<size;i++)
		printf("%d - %.3lf\n",i,vector[i]);
}

int main(int argc, char *argv[]){

	
	clock_t start,stop;
	int k,j,i,multiply;
	FILE * pFile;
	size_t result;

	//UPDATE1 :now P and Q are 1D to match cuda
	my_float *P;		//Buneman Series 
	my_float *Q;		//Buneman Series
	//UPDATE 2 : now B,T,F,X are 2D but malloc is sequencial 1D to match CUDA
	my_float *B;
	my_float *T; 
	my_float *F;
	my_float *X;
	int n ,q;
	n = atoi(argv[1]);
	q = atoi(argv[2]);
	int jq = (int)log2(n+1);
	malloc1D(&P,jq*n*q);
	malloc1D(&Q,jq*n*q);
	malloc1D(&T,q*q);
	malloc1D(&B,q*q);
	malloc1D(&F,n*q);
	malloc1D(&X,n*q);


	// allocate device memory
	my_float  *d_P, *d_Q, *d_T,*d_B, *d_F,*d_X;
	my_float *d_tempMatrix[3],*d_tempVector[3],*d_matrixT,*d_matrixB;
	int *ipiv;

	cudaMalloc((void **) &d_P, 2*q*q*sizeof(my_float));
	cudaMalloc((void **) &d_Q, 2*q*q*sizeof(my_float));
	cudaMalloc((void **) &d_T, q*q*sizeof(my_float));
	cudaMalloc((void **) &d_B, q*q*sizeof(my_float));
	//cudaMalloc((void **) &d_F, n*q*sizeof(my_float));
	cudaMalloc((void **) &d_X, n*q*sizeof(my_float));

	cudaMalloc((void **) &d_tempMatrix[0], q*q*sizeof(my_float));
	cudaMalloc((void **) &d_tempMatrix[1], q*q*sizeof(my_float));
	cudaMalloc((void **) &d_tempMatrix[2], q*q*sizeof(my_float));
	cudaMalloc((void **) &d_tempVector[0], q*sizeof(my_float));
	cudaMalloc((void **) &d_tempVector[1], q*sizeof(my_float));
	cudaMalloc((void **) &d_tempVector[2], q*sizeof(my_float));

	cudaMalloc((void **) &d_matrixT, q*q*sizeof(my_float));
	cudaMalloc((void **) &d_matrixB, q*q*sizeof(my_float));
	cudaMalloc(&ipiv,(q+1)*sizeof(int));

	culaStatus s;
	s = culaInitialize();
	if(s != culaNoError){
		printf("%s\n", culaGetStatusString(s));
		exit(1);
	}
	cublasInit();
	cublasCreate(&handle);	

	for(i=0;i<q;i++){
		for(j=0;j<q;j++){
			if(i==j){
				B[i*q+j] =  4.0;
				T[i*q+j] = -1.0;
			}
			else if(i ==j+1 || j==i+1){
				B[i*q+j] = -1.0;
				T[i*q+j] = 0.0;
			}
			else {
				B[i*q+j] = 0.0;
				T[i*q+j] = 0.0;
			}
		}
	}


	for(i=0;i<n;i++){
		for(j=0;j<q;j++){
			F[i*q+j] = 1.0;
			X[i*q+j] = 0.0;
		}
	}

	cudaMemset(d_Q,0,2*n*q*sizeof(my_float));
	//Part 1
	for(i=0;i<n;i++){
		for(j=0;j<q;j++){
			P[i3to1(0,i,j,n,q)] = 0.0;
			Q[i3to1(0,i,j,n,q)] = F[i*q+j];
		}
	} 

	start = clock();

	cudaMemcpy(d_B, B, q*q*sizeof(my_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, T, q*q*sizeof(my_float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_F, F, n*q*sizeof(my_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_X, X, n*q*sizeof(my_float), cudaMemcpyHostToDevice);	
	//cudaMemcpy(d_P, P, jq*q*q*sizeof(my_float), cudaMemcpyHostToDevice);
	
	
	
	if(T[0]<0) multiply = 1;
	else multiply = -1;

	for(k=1;k<=jq-1;k++){
		//printf("k = %d\n",k);
		calcT(d_matrixT,d_tempMatrix[2],d_T,q,k-1);
		calcInvB(d_matrixB,d_B,d_T,q,k-1,multiply,ipiv,d_tempMatrix[0]);

		cudaMemcpy(&d_Q[i3to1(0,0,0,n,q)], &Q[i3to1(k-1,0,0,n,q)], n*q*sizeof(my_float), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_P[i3to1(0,0,0,n,q)], &P[i3to1(k-1,0,0,n,q)], n*q*sizeof(my_float), cudaMemcpyHostToDevice);
		
		for(j=0;j<pow(2,jq-k)-1;j++){
			int idx1 = pow(2,k)*(j+1)-1;
			int idx2 = pow(2,k-1);

			vector_addition(d_tempVector[0],&d_P[i3to1(0,idx1-idx2,0,n,q)],&d_P[i3to1(0,idx1+idx2,0,n,q)],q,q,1,1);
			matrixVector_multiplication(d_tempVector[1],d_matrixT,d_tempVector[0],q,q,q,1.0);
			vector_addition(d_tempVector[0],d_tempVector[1],&d_Q[i3to1(0,idx1,0,0,q)],q,q,1,-1);
			
			matrixVector_multiplication(d_tempVector[1],d_matrixB,d_tempVector[0],q,q,q,1.0);
			vector_addition(&d_P[i3to1(1,idx1,0,n,q)],&d_P[i3to1(0,idx1,0,n,q)],d_tempVector[1],q,q,1,-1);
			vector_addition(d_tempVector[0],&d_Q[i3to1(0,idx1-idx2,0,n,q)],&d_Q[i3to1(0,idx1+idx2,0,n,q)],q,q,1,1);
			
			matrixVector_multiplication(d_tempVector[1],d_matrixT,&d_P[i3to1(1,idx1,0,n,q)],q,q,q,2.0);
			vector_addition(d_tempVector[0],d_tempVector[0],d_tempVector[1],q,q,1,-1);
			
			matrixVector_multiplication(&d_Q[i3to1(1,idx1,0,n,q)],d_matrixT,d_tempVector[0],q,q,q,1.0);
		}
		cudaMemcpy(&Q[i3to1(k,0,0,n,q)],&d_Q[i3to1(1,0,0,n,q)],n*q*sizeof(my_float),cudaMemcpyDeviceToHost);
		cudaMemcpy(&P[i3to1(k,0,0,n,q)],&d_P[i3to1(1,0,0,n,q)],n*q*sizeof(my_float),cudaMemcpyDeviceToHost);
	}
	
	
	//Part2
	int idx1 = pow(2,k) - pow(2,k-1) -1;
	calcInvB(d_matrixB,d_B,d_T,q,k-1,multiply,ipiv,d_tempMatrix[0]);
	matrixVector_multiplication(d_tempVector[0],d_matrixB,&d_Q[i3to1(1,idx1,0,n,q)],q,q,q,1.0);
	vector_addition(&d_X[idx1*q],d_tempVector[0],&d_P[i3to1(1,idx1,0,n,q)],q,q,1,1);
	
	//Part3
	for(k = jq-1;k>=1;k--){
		//printf("k = %d\n",k);
		int point;
		calcT(d_matrixT,d_tempMatrix[2],d_T,q,k-1);
		calcInvB(d_matrixB,d_B,d_T,q,k-1,multiply,ipiv,d_tempMatrix[0]);
		cudaMemcpy(&d_Q[i3to1(0,0,0,n,q)], &Q[i3to1(k-1,0,0,n,q)], n*q*sizeof(my_float), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_P[i3to1(0,0,0,n,q)], &P[i3to1(k-1,0,0,n,q)], n*q*sizeof(my_float), cudaMemcpyHostToDevice);

		for(j= 1;j<(int)pow(2,jq-k)-1;j++){
			point = pow(2,k)*(j+1)-pow(2,k-1)-1;
			vector_addition(d_tempVector[0],&d_X[(int)(pow(2,k)*(j+1)-1)*q],  &d_X[(int)(pow(2,k)*(j+1) - pow(2,k)-1)*q],q,q,1,1);
			matrixVector_multiplication(d_tempVector[1],d_matrixT,d_tempVector[0],q,q,q,1.0);
			vector_addition(d_tempVector[0],&d_Q[i3to1(0,point,0,n,q)],d_tempVector[1],q,q,1,-1);
			matrixVector_multiplication(d_tempVector[1],d_matrixB,d_tempVector[0],q,q,q,1.0);
			vector_addition(&d_X[point*q],d_tempVector[1],&d_P[i3to1(0,point,0,n,q)],q,q,1,1);
		}
		j = 0;
		point = pow(2,k) -pow(2,k-1)-1;
		matrixVector_multiplication(d_tempVector[0],d_matrixT,&d_X[(int)(pow(2,k)-1)*q],q,q,q,1.0);
		vector_addition(d_tempVector[1],&d_Q[i3to1(0,point,0,n,q)],d_tempVector[0],q,q,1,-1);
		matrixVector_multiplication(d_tempVector[0],d_matrixB,d_tempVector[1],q,q,q,1.0);
		vector_addition(&d_X[point*q],d_tempVector[0],&d_P[i3to1(0,point,0,n,q)],q,q,1,1);

		j = (int)(pow(2,jq-k)-1);
		point = pow(2,jq)-pow(2,k-1)-1; 
		matrixVector_multiplication(d_tempVector[0],d_matrixT,&d_X[(int)(pow(2,jq)-pow(2,k)-1)*q],q,q,q,1.0);
		vector_addition(d_tempVector[1],&d_Q[i3to1(0,point,0,n,q)],d_tempVector[0],q,q,1,-1);
		matrixVector_multiplication(d_tempVector[0],d_matrixB,d_tempVector[1],q,q,q,1.0);
		vector_addition(&d_X[point*q],d_tempVector[0],&d_P[i3to1(0,point,0,n,q)],q,q,1,1);
	}
	cudaMemcpy(X,d_X,n*q*sizeof(my_float),cudaMemcpyDeviceToHost);
	stop = clock();

	printf("%d\t%d\t%.3lf\n",n,q,(my_float)(stop-start)/CLOCKS_PER_SEC/4);
	cublasDestroy(handle);  
	cublasShutdown();
	culaShutdown();
	cudaDeviceReset();


#ifdef verbose
	printf("\nprint_matrix()\n");
	print_vector(X,n*q);
#endif
	return 1;

}
