/*
*	Buneman's algorithm for block tridiagonal systems 
*	This is a variation of block cyclic reduction that improves stability
*	MKL library is used for matrix operations
*   CPU Implementation
*	minavouronikoy@gmail.com - elzisiou@gmail.com
*   Master thesis
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <string.h>
#include <time.h>

#include <mkl.h>
#include <mkl_scalapack.h>

#include "../global.h"
#include "matrix.h"
//#include "memory.h"

#define PI 3.14159265358979323846

//int n = Ni;
//int q = Qi;

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
	memcpy(T1,T2,sizeof(my_float)*size*size);
	memcpy(Tres,T2,sizeof(my_float)*size*size);

	int exponent = pow(2,index);
	for(p=0;p<exponent-1;p++){

		matrix_multiplication(Tres,T1,T2,size,size,size,size);
		memcpy(T1,Tres,sizeof(my_float)*size*size);
	}
	return 1;
}

int calcInvB(my_float *add,my_float * B,my_float * T,int size,int index,int multiply){
	int i,j,l;
	my_float theta,cosine,alpha;
	my_float *temp[2];
	int *p;
	int lwork = size*size;
	my_float *work;

	temp[0] = (my_float*)mkl_malloc( sizeof(my_float)*size*size, 128 );
	temp[1] = (my_float*)mkl_malloc( sizeof(my_float)*size*size, 128 );
	p = (int *)mkl_malloc(sizeof(int)*size,64);
	work = (my_float *)mkl_malloc(sizeof(my_float)*lwork,128);

	int to = pow(2,index);

	for(i=0;i<size;i++)
		for(j=0;j<size;j++)
			add[i*size+j] = 0.0;

	for(l=1;l<=to;l++){
		theta = calc_theta(index,l);
		cosine = -2*cos(theta);
		alpha = calc_alpha(index,l,theta);

		matrix_addition(temp[1],B,T,size,size,size,size,1,cosine*multiply);
		inverse_matrix(temp[1],p,work,lwork,size);  
		matrix_addition(add,add,temp[1],size,size,size,size,1,-alpha);

	}
	mkl_free(temp[0]);
	mkl_free(temp[1]);
	mkl_free(p);
	mkl_free(work);
	return 1;
}

int main(int argc,char *argv[]){
	int jq; 
	clock_t start,stop;
	int k,j,i,multiply,n,q;
	my_float *tempVector[3];
	my_float *tempMatrix[3];
	my_float *matrixT,*matrixB;
	FILE * pFile;
	my_float *P;		//Buneman Series 
	my_float *Q;		//Buneman Series
	my_float *B;
	my_float *T; 
	my_float *F;
	my_float *X;
	n = atoi(argv[1]);
	q = atoi(argv[2]);
	jq = (int)log2(n+1);
	//allocate memory

    tempVector[0] = (my_float*)mkl_malloc( sizeof(my_float)*q, 64 );
	tempVector[1] = (my_float*)mkl_malloc( sizeof(my_float)*q, 64 );
	tempVector[2] = (my_float*)mkl_malloc( sizeof(my_float)*q, 64 );
	tempMatrix[0] = (my_float*)mkl_malloc( sizeof(my_float)*q*q, 64 );
	tempMatrix[1] = (my_float*)mkl_malloc( sizeof(my_float)*q*q, 64 );
	tempMatrix[2] = (my_float*)mkl_malloc( sizeof(my_float)*q*q, 64 );
	matrixT = (my_float*)mkl_malloc( sizeof(my_float)*q*q, 64 );
    matrixB = (my_float*)mkl_malloc( sizeof(my_float)*q*q, 64 );
	P = (my_float*)mkl_malloc( sizeof(my_float)*jq*n*q, 64 );
	Q = (my_float*)mkl_malloc( sizeof(my_float)*jq*n*q, 64 );
	B = (my_float*)mkl_malloc( sizeof(my_float)*q*q, 64 );
	T = (my_float*)mkl_malloc( sizeof(my_float)*q*q, 64 );
	 F= (my_float*)mkl_malloc( sizeof(my_float)*n*q, 64 );
	X= (my_float*)mkl_malloc( sizeof(my_float)*n*q, 64 );

	//Initialize data
	for(i=0;i<q;i++){
		for(j=0;j<q;j++){
			if(i==j){
				B[i*q+j] = 4.0;
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

	for(i=0;i<n;i++){
		for(j=0;j<q;j++){
			P[i3to1(0,i,j,n,q)] = 0.0;
			Q[i3to1(0,i,j,n,q)] = F[i*q+j];			
		}
	}

   //Start measuring time
	start = clock();

	//Part 1 - Forward Reduction
	if(T[0*q+0]<0) multiply = 1;
	else multiply = -1;
	for(k=1;k<=jq-1;k++){
		//printf("k = %d\n",k);
		calcT(matrixT,tempMatrix[2],T,q,k-1);
		calcInvB(matrixB,B,T,q,k-1,multiply);
		for(j=0;j<pow(2,jq-k)-1;j++){
			int idx1 = pow(2,k)*(j+1)-1;
			int idx2 = pow(2,k-1);
			vector_addition(tempVector[0],&P[i3to1(k-1,idx1-idx2,0,n,q)],&P[i3to1(k-1,idx1+idx2,0,n,q)],q,q,1,1);
			
			matrixVector_multiplication(tempVector[1],matrixT,tempVector[0],q,q,q,1.0);
			
			vector_addition(tempVector[0],tempVector[1],&Q[i3to1(k-1,idx1,0,n,q)],q,q,1,-1);
			
			matrixVector_multiplication(tempVector[1],matrixB,tempVector[0],q,q,q,1.0);	
			
			vector_addition(&P[i3to1(k,idx1,0,n,q)],&P[i3to1(k-1,idx1,0,n,q)],tempVector[1],q,q,1,-1);	
		
			vector_addition(tempVector[0],&Q[i3to1(k-1,idx1-idx2,0,n,q)],&Q[i3to1(k-1,idx1+idx2,0,n,q)],q,q,1,1);		
				
			matrixVector_multiplication(tempVector[1],matrixT,&P[i3to1(k,idx1,0,n,q)],q,q,q,2.0);			
			vector_addition(tempVector[0],tempVector[0],tempVector[1],q,q,1,-1);			
			matrixVector_multiplication(&Q[i3to1(k,idx1,0,n,q)],matrixT,tempVector[0],q,q,q,1.0);
			
		}
	}

	//Part2 - Solve one sustem
	int idx1 = pow(2,k) - pow(2,k-1) -1;
	calcInvB(matrixB,B,T,q,k-1,multiply);
	matrixVector_multiplication(tempVector[0],matrixB,&Q[i3to1(k-1,idx1,0,n,q)],q,q,q,1.0);
	vector_addition(&X[idx1*q],tempVector[0],&P[i3to1(k-1,idx1,0,n,q)],q,q,1,1);

	//Part3 - Backward Substitution
	for(k = jq-1;k>=1;k--){
		//printf("k = %d\n",k);
		int point;
		calcT(matrixT,tempMatrix[2],T,q,k-1);
		calcInvB(matrixB,B,T,q,k-1,multiply);
		for(j= 1;j<(int)pow(2,jq-k)-1;j++){
			point = pow(2,k)*(j+1)-pow(2,k-1)-1;
			vector_addition(tempVector[0],&X[(int)(pow(2,k)*(j+1)-1)*q],  &X[(int)(pow(2,k)*(j+1) - pow(2,k)-1)*q],q,q,1,1);
			matrixVector_multiplication(tempVector[1],matrixT,tempVector[0],q,q,q,1.0);
			vector_addition(tempVector[0],&Q[i3to1(k-1,point,0,n,q)],tempVector[1],q,q,1,-1);
			matrixVector_multiplication(tempVector[1],matrixB,tempVector[0],q,q,q,1.0);
			vector_addition(&X[point*q],tempVector[1],&P[i3to1(k-1,point,0,n,q)],q,q,1,1);
		}
		j = 0;
		point = pow(2,k) -pow(2,k-1)-1;
		matrixVector_multiplication(tempVector[0],matrixT,&X[(int)(pow(2,k)-1)*q],q,q,q,1.0);
		vector_addition(tempVector[1],&Q[i3to1(k-1,point,0,n,q)],tempVector[0],q,q,1,-1);
		matrixVector_multiplication(tempVector[0],matrixB,tempVector[1],q,q,q,1.0);
		vector_addition(&X[point*q],tempVector[0],&P[i3to1(k-1,point,0,n,q)],q,q,1,1);


		j = (int)(pow(2,jq-k)-1);
		point = pow(2,jq)-pow(2,k-1)-1; 
		matrixVector_multiplication(tempVector[0],matrixT,&X[(int)(pow(2,jq)-pow(2,k)-1)*q],q,q,q,1.0);
		vector_addition(tempVector[1],&Q[i3to1(k-1,point,0,n,q)],tempVector[0],q,q,1,-1);
		matrixVector_multiplication(tempVector[0],matrixB,tempVector[1],q,q,q,1.0);
		vector_addition(&X[point*q],tempVector[0],&P[i3to1(k-1,point,0,n,q)],q,q,1,1);

	}

	stop = clock();
	printf("\nn = %d\tq =%d\tDuration %.3lf\n",n,q,(my_float)(stop-start)/CLOCKS_PER_SEC/4);
	
/* Check the correctness of the results */
#ifdef checkResults
	pFile = fopen ("myfile.bin", "wb");
	fwrite (X , sizeof(my_float), n*q, pFile);
	fclose (pFile);
#endif
//print out the results
#ifdef verbose
	printf("\nprint_matrix()\n");
	print_vector(X,n*q);
#endif
	return 1;

}
