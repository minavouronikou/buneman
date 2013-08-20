icc -Wall  -O2 block.cpp matrix.cpp memory.cpp -o cr -mkl:parallel
echo "Start experiments on CPU"

./cr 2047 2047 >> results.txt
./cr 4095 4095 >> results.txt

echo "Finish experiments on CPU"
