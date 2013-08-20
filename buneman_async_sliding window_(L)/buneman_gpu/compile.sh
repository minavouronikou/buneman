nvcc -arch=sm_30 -O3 block.cu matrix.cu memory.cu -o cr -I$CULA_INC_PATH -L$CULA_LIB_PATH_64 -lcula_lapack -lcublas -lcudart -m64
echo "Start experiments on GPU"
./cr 1023 1023
echo "Finish experiments on GPU"
