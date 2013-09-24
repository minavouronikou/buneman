A block cyclic reduction algorithm was implemented for CPU and ported to GPU to study the performance.
The CPU code was implemented in C and Matlab. The GPU code was implemented in NVIDIA CUDA. 
Due to numerically instability of Block Cyclic Reduction the Buneman variant was implemented. 
Extensive experiments were performed and their results were compared and studied. 
It was proved that in GPUs the Buneman algorithm shows up to 25x speedup in relation to CPU.
 That concludes that GPU can be used to solve efficiently, accurately and clearly
faster these problems especially in two dimensions were the speedup was particularly
remarkable and the computational load extremely large.

MKL implementations for Blas and Lapack were used on CPU
CUBLAS and CULapack were used for Blas operations on GPU

*Buneman_N_q_(L) is the first implementation containing CPU and GPU code

*Buneman (solver)(L) is a CPU implementation were was attempted to replace the inversion of matrices
with a linear system solution due to high cost of inversion

*Buneman_sliding_window_linux is optimization which came of
the need to examine larger problems. Storage of Q and P matrices, that consist the
Buneman Series, are the most spatial expensive. A ”sliding window” technique was
used. Instead of transferring all the matrices P and Q on GPU, it is preferable to load
in each step two matrices to hold the current and the previous iteration. Although
a lot of space is now saved and larger problems can be executed, additional memory
transfers between host and device must be done.


*Buneman_async_sliding_window_linux  is an optimization of the above.Streams were used in order
to overlap computation with memory transfers.

All projects have a compile.sh bash script for compilation and for run of some experiments. In the run of the
executable file the numbers are the N,Q respectively.


