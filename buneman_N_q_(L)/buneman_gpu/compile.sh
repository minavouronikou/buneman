nvcc -arch=sm_30 -O3 block.cu matrix.cu memory.cu -o cr -I$CULA_INC_PATH -L$CULA_LIB_PATH_64 -lcula_lapack -lcublas -lcudart
echo "Start experiments on GPU"
./cr 3 127 > results.txt
./cr 3 255 >> results.txt
./cr 3 511 >> results.txt
./cr 3 1023 >> results.txt
./cr 3 2047 >> results.txt
./cr 3 4095 >> results.txt

./cr 7 127 >> results.txt
./cr 7 255 >> results.txt
./cr 7 511 >> results.txt
./cr 7 1023 >> results.txt
./cr 7 2047 >> results.txt
./cr 7 4095 >> results.txt

./cr 15 127 >> results.txt
./cr 15 255 >> results.txt
./cr 15 511 >> results.txt
./cr 15 1023 >> results.txt
./cr 15 2047 >> results.txt
./cr 15 4095 >> results.txt

./cr 31 127 >> results.txt
./cr 31 255 >> results.txt
./cr 31 511 >> results.txt
./cr 31 1023 >> results.txt
./cr 31 2047 >> results.txt
./cr 31 4095 >> results.txt

./cr 63 127 >> results.txt
./cr 63 255 >> results.txt
./cr 63 511 >> results.txt
./cr 63 1023 >> results.txt
./cr 63 2047 >> results.txt
./cr 63 4095 >> results.txt

./cr 127 127 >> results.txt
./cr 127 255 >> results.txt
./cr 127 511 >> results.txt
./cr 127 1023 >> results.txt
./cr 127 2047 >> results.txt
./cr 127 4095 >> results.txt

./cr 255 255 >> results.txt 
./cr 511 511 >> results.txt
./cr 1023 1023 >> results.txt
./cr 2047 2047 >> results.txt
./cr 4095 4095 >> results.txt
echo "Finish experiments on GPU"
