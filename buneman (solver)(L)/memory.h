#include "global.h"

int malloc1D(my_float **p,int n);
int malloc2D(my_float ***p,int n,int m);
int malloc2D_s(my_float ***p,int n,int m);
int malloc3D(my_float ****p,int n,int m,int z);
void free2D(my_float **p,int n);
void free2D_s(my_float **p,int n);
void free3D(my_float ***p,int n,int m);
