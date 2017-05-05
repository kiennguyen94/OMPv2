#include <stdio.h>
#include <stdlib.h>
//#include "cgls.cuh"
#include <cuda.h>
#include <iostream>
#include <fstream>
#include "magma_lapack.h"
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "magma_v2.h"
#include <cublas_v2.h>
#include <curand.h>
#include <armadillo>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "OMP_alt.h"

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

template<typename T>
void printarr(T* arr, const int N) {
	for (int i = 0; i < N; i++) {
		std::cout << arr[i] << ' ';
	}
	std::cout << '\n';
}

//double second(void) {
//	struct timeval tv;
//	gettimeofday(&tv, NULL);
//	return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
//}

void GPU_fill_rand(float* A, int numel, bool isNormal) {
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	if (isNormal == true) {
		curandGenerateNormal(prng, A, numel, 0, 1);
	} else {
		curandGenerateUniform(prng, A, numel);
	}
}

int flush(cublasHandle_t handle, float* in, int size) {
	float alpha = 0.0;
	// std::cout<<"While flushing\n";
	cublasStatus_t stat = cublasSscal(handle, size, &alpha, in, 1);
	// std::cout<<"While flushing1\n";
	if (stat != CUBLAS_STATUS_SUCCESS) {
		return 1;
	}
	return 0;
}

// Calculate Mean square error
float MSE(float *a, float *b, int size) {
	float result = 0.0;
	int i = 0;

#pragma omp parallel for \
        default(shared) private(i) \
        reduction(+:result)

	for (i = 0; i < size; i++) {
		result += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return result / float(size);
}

#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

/* Least square using QR decomposition
 * Solve ||Ax-b||
 * A: M*N
 * x: N*1
 * b: M*1
 * */
void LSQR(magmaFloat_ptr A_d, magmaFloat_ptr b_d, magmaFloat_ptr x_d, int M,
		int N, magma_queue_t queue) {
	magmaFloat_ptr A_copy, b_copy;
	float *h_work;
	magma_int_t inc = 1;
	magma_int_t lda, ldb, lhwork, lworkgpu, nrhs, info;
	nrhs = 1;
	lda = M;
	ldb = M;
	magma_int_t nb = magma_get_sgeqrf_nb(M, N);
	lworkgpu = (M - N + nb) * (nrhs + nb) + nrhs * nb;
	lhwork = lworkgpu;
	TESTING_CHECK(magma_smalloc(&A_copy, M * N));
	TESTING_CHECK(magma_smalloc(&b_copy, M));
	TESTING_CHECK(magma_smalloc_cpu(&h_work, lhwork));

	magma_scopymatrix_async(M, N, A_d, M, A_copy, M, queue);
	magma_scopyvector_async(M, b_d, inc, b_copy, inc, queue);

	magma_sgels_gpu(MagmaNoTrans, M, N, nrhs, A_copy, lda, b_copy, ldb, h_work,
			lworkgpu, &info);

	magma_scopyvector(N, b_copy, inc, x_d, inc, queue);

	magma_free(A_copy);
	magma_free(b_copy);

}

void TestLSQR() {
	int A_row = 200;
	int A_col = 100;
	float *A_d, *x_d, *b_d, *x_hat_d;
	float *x_h = new float[A_col];
	float *xhat_h = new float[A_col];

	cudaMalloc(&A_d, sizeof(float) * A_row * A_col);
	cudaMalloc(&x_d, sizeof(float) * A_col);
	cudaMalloc(&b_d, sizeof(float) * A_row);
	GPU_fill_rand(A_d, A_row * A_col, false);
	GPU_fill_rand(x_d, A_col, false);
	cudaMemcpy(x_h, x_d, sizeof(float) * A_col, cudaMemcpyDeviceToHost);

	cublasStatus_t mat_vect;
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0;
	float beta = 0.0;
	// Generate b. Now b_h has content of array b
	mat_vect = cublasSgemv(handle, CUBLAS_OP_N, A_row, A_col, &alpha, A_d,
			A_row, x_d, 1, &beta, b_d, 1);
	if (flush(handle, x_d, A_col) != CUBLAS_STATUS_SUCCESS) {
		std::cout << "Unable to flush x_d\n";
	}
	magma_queue_t queue = NULL;
	magma_int_t dev = 0;
	magma_queue_create(dev, &queue);

	LSQR(A_d, b_d, x_d, A_row, A_col, queue);
	cudaMemcpy(xhat_h, x_d, sizeof(float) * A_col, cudaMemcpyDeviceToHost);
	float mse = MSE(xhat_h, x_h, A_col);
	cout << "MSE " << mse << endl;

}

/* Orthogonal Matchin Pursuit
 * A: dictionary, N * M
 * b: observed signal, N * 1
 * coeff: reconstructed signal, M * 1
 * k: sparsity
 * row: N
 * col: M
 *
 * All inputs are host array
 * */

void OMP(const float* A, const float* b, float* coeff, int k, int N, int M) {
	magma_int_t info = 0;
	magma_queue_t queue = NULL;
	magma_int_t dev = 0;
	magma_int_t incx = 1;
	magma_queue_create(dev, &queue);
	float alpha = 1.0;
	float beta = 0.0;

	// Init memory
	magmaFloat_ptr A_d, b_d, res_d, dot_p, A_ptr, temp, B, u1, u2, F, d;
	float *atom_d, *atom_ptr;
	magmaFloat_ptr x_T;
	float * x_T_h;
	cudaMallocManaged(&atom_d, N * k * sizeof(float));
	TESTING_CHECK(magma_smalloc_cpu(&x_T_h, k));
	TESTING_CHECK(magma_smalloc(&x_T, k));
	TESTING_CHECK(magma_smalloc(&A_d, N * M));
	TESTING_CHECK(magma_smalloc(&b_d, N));
	TESTING_CHECK(magma_smalloc(&res_d, N));
	TESTING_CHECK(magma_smalloc(&dot_p, M));
	// N*K because after k iterations we get at most k columns
//	TESTING_CHECK(magma_smalloc(&atom_d, N * k));
	TESTING_CHECK(magma_smalloc(&temp, N));

	// TODO: fill this
	//	TESTING_CHECK(magma_smalloc(&temp, ));
	//	TESTING_CHECK(magma_smalloc(&x_T, ));

	// Copy from host
	(magma_ssetmatrix(N, M, A, N, A_d, N, queue));
	(magma_ssetvector(N, b, 1, b_d, 1, queue));
	(magma_scopyvector(N, b_d, 1, res_d, 1, queue));
//	magma_sprint_gpu(N, M, A_d, N, queue);
	// Test Code
//	test_ptr = A_d;
//	test_ptr += 10;
//	magma_sprint_gpu(20, 1, A_d, 20,  queue);
//	magma_sprint_gpu(10,1, test_ptr, 10, queue);
//	cout<<*test_ptr<<endl;

	thrust::device_vector<int> indx_d(k, 0);
	thrust::host_vector<int> indx(k, 0);
	atom_ptr = atom_d;
	A_ptr = A_d;
	for (int i = 0; i < k; i++) {
		(magma_sgemv(MagmaTrans, N, M, alpha, A_d, N, res_d, incx, beta, dot_p,
				incx, queue));
//		magma_sprint_gpu(M, 1, dot_p, M, queue);
		magma_int_t max_index = magma_isamax(M, dot_p, incx, queue)-1;
//		cout<<max_index<<endl;
		indx_d[i] = max_index;
		// A_ptr points to the start of the column from which we are copying
		A_ptr = A_d;
		A_ptr += max_index * N;
		magma_scopyvector(N, A_ptr, incx, atom_ptr, incx, queue);
		atom_ptr += N;
		// temporary result from least square inversion

		/* atom_d: N * (i+1), i \in [1:k]
		 * b: N * 1
		 * x_T: (i+1) * 1
		 * */
//		magma_sprint_gpu(N, i+1, atom_d, N, queue);
		LSQR(atom_d, b_d, x_T, N, i + 1, queue);
//		magma_sprint_gpu(i+1, 1, x_T, i+1, queue);
		// temp: N * 1
		magma_sgemv(MagmaNoTrans, N, i + 1, alpha, atom_d, N, x_T, incx, beta,
				temp, incx, queue);
//		magma_sprint_gpu(N, 1, temp, N, queue);

		(magma_scopyvector(N, b_d, 1, res_d, 1, queue));

		magma_saxpy(N, -1.0, temp, incx, res_d, incx, queue);
//		break;
	}
//	magma_sprint_gpu(k, 1, x_T, k, queue);
	// Copy back to host
	indx = indx_d;
	magma_sgetvector(k, x_T, 1, x_T_h, 1, queue);

# pragma omp parallel for
	for (int i = 0; i < k; i++) {
		coeff[indx[i]] = x_T_h[i];
	}

	std::cout << "return sucessfully!\n";
}

using namespace arma;
using namespace std;
int main() {
	int N = 2048;
	int M = 16384;
	int K = 256;

	string A_name = "A.bin";
	string b_name = "b.bin";
	string x_name = "x.bin";
	mat A;
	A.load(A_name, arma::raw_binary);
//	A.print();
//	arma::inplace_trans(A);
	mat b;
	b.load(b_name, arma::raw_binary);
	mat x_mat;
	x_mat.load(x_name, arma::raw_binary);

	fmat A_ = conv_to<fmat>::from(A);
	fmat b_ = conv_to<fmat>::from(b);
	fmat x_ = conv_to<fmat>::from(x_mat);

	float* A_ptr = A_.memptr();
	float* b_ptr = b_.memptr();
	float* x_true = x_.memptr();
	float* x = new float[M];
	double start, stop;
	magma_init();
	start = second();
//	OMP(A_ptr, b_ptr, x, K, N, M);
	OMP_alt(A_ptr, b_ptr, x, K, N, M);
	stop = second();
//	OMP_alt(A_ptr, b_ptr, x, K, N, M);

	float mse = MSE(x_true, x, M);

	cout<<mse<<endl;
//	cout<<stop-start<<endl;
//	TestLSQR();

//	test();
//	OMP_alt(A_ptr, b_ptr, x, K, N, M);

	magma_finalize();
	helloworld();

}
