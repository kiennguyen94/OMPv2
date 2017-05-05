# OMP

Implementation of orthogonal matching pursuit in GPU.
Requires Cublas, MAGMA to run.

The implementation is in src/OMP_alt.cu. The test code is in OMP.cu, which requires 3 .bin files that are generated from matlab.
Just note that matrix A has to be normalized: A = A ./ sqrt(sum(A.^2, 1)). The function returns result in array x, stored in the RAM.
