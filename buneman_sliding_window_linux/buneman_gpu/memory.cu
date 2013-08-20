#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "memory.h"

#ifdef MKL
int malloc2D_s(my_float ***p,int n,int m){
	int i;
	*p = (my_float **)malloc(sizeof(my_float *)*n);
	if(*p==NULL){
		printf("Cannot malloc memory 2d_s\n");
		exit(0);
	}
	(*p)[0] = (my_float *)malloc(sizeof(my_float)*n*m);

	if((*p)[0]==NULL){
		printf("Cannot malloc memory 2d_s\n");
		exit(0);
	}

	for(i=1;i<n;i++){
		(*p)[i] = (*p)[i-1] + m;
	}
	return 1;
}

void free2D_s(my_float **p,int m){
	int i;
	free(p[0]);
	free(p);
}

#else 
void free2D_s(my_float **p,int m){
	free(p[0]);
	free(p);
}

int malloc2D_s(my_float ***p,int n,int m){
	int i;
	*p = (my_float **)malloc(sizeof(my_float *)*n);
	if(*p==NULL){
		printf("Cannot malloc memory 2d_s\n");
		exit(0);
	}
	(*p)[0] = (my_float *)malloc(sizeof(my_float)*n*m);

	if((*p)[0]==NULL){
		printf("Cannot malloc memory 2d_s\n");
		exit(0);
	}

	for(i=1;i<n;i++){
		(*p)[i] = (*p)[i-1] + m;
	}
	return 1;
}
#endif

int malloc1D(my_float **p,int z){
	*p = (my_float *)malloc(sizeof(my_float)*z);
	if(*p==NULL){
		printf("Cannot malloc memory 1d\n");
		exit(0);
	}

	return 1;
}

int malloc2D(my_float ***p,int m,int z){
	int i;
	*p = (my_float **)malloc(sizeof(my_float *)*m);

	if(*p == NULL)
		return 0;

	for(i=0;i<m;i++){
		if(!malloc1D(&(*p)[i],z)) return 0;
	}

	return 1;
}

int malloc3D(my_float ****p,int n,int m,int z){
	int i;
	*p = (my_float ***)malloc(sizeof(my_float *)*n);
	if(*p == NULL)
		return 0;

	for(i=0;i<n;i++){
		if(!malloc2D(&(*p)[i],m,z)) return 0;
	}
	return 1;
}



void free2D(my_float **p,int m){
	int i;
	for(i=0;i<m;i++){
		free(p[i]);
	}
	free(p);
}

void free3D(my_float ***p,int n,int m){
	int i;
	for(i=0;i<n;i++){
		free2D(p[i],m);
	}
	free(p);
}