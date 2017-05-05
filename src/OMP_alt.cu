#include "OMP_alt.h"
void helloworld(){
	std::cout<<"hi world\n";
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
double second(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
}

void OMP_alt(const float* A, const float* b, float* coeff, int k, int N, int M){

	// Timer
	double start, stop;

	// Init Memory
	magma_int_t incx = 1;
	magma_queue_t queue = NULL;
	magma_int_t dev = 0;
	magma_queue_create(dev, &queue);

	magmaFloat_ptr A_d, b_d, res_d, B,  u1, u2, F, d, atom_d, x_T, dot_p, temp, u;
	magmaFloat_ptr A_ptr, B_row_ptr, B_col_ptr, atom_ptr;

	float alpha = 1.0;
	float beta = 0.0;

	float *x_T_h;


	// Allocate mem
	magma_smalloc_cpu(&x_T_h, k);
	magma_smalloc(&A_d, N*M);
	magma_smalloc(&b_d, N);
	magma_smalloc(&res_d, N);
	magma_smalloc(&dot_p, M);
	magma_smalloc(&temp, k);
	magma_smalloc(&atom_d, N*k);
	magma_smalloc(&B, k * k);
	magma_smalloc(&u, k);
	magma_smalloc(&u1, k);
	magma_smalloc(&u2, k);

	// Copy from host
	magma_ssetmatrix(N, M, A, N, A_d, N, queue);

	magma_ssetvector(N, b, 1, b_d, 1, queue);
	magma_ssetvector(N, b, 1, res_d, 1, queue);
	start = second();

//	thrust::device_vector<int> indx_d(k,0);
	thrust::host_vector<int> indx(k,0);

	atom_ptr = atom_d;
	A_ptr = A_d;
	// First iteration
	magma_sgemv(MagmaTrans, N, M, alpha, A_d, N, res_d, incx, beta, dot_p, incx, queue);
	magma_int_t max_index = magma_isamax(M, dot_p, incx, queue) - 1;
	indx[0] = max_index;
	A_ptr += max_index * N;
	magma_scopyvector(N, A_ptr, incx, atom_ptr, incx, queue);
	atom_ptr += N;
	// Calculate B
	float B_temp = magma_sdot(N, atom_d, 1, atom_d, 1, queue);
	B_temp = 1.0 / B_temp;
	float *B_temp_ptr = &B_temp;
	magma_ssetvector(1, B_temp_ptr, 1, B, 1, queue);

	// Calculate u
	// Atoms.T.dot(y)
	float Aty = magma_sdot(N, atom_d, 1, b_d, 1, queue);
	float u_temp = B_temp * Aty;
	float *u_temp_ptr = &u_temp;
	magma_ssetvector(1, u_temp_ptr, 1, u, 1, queue);

	// res = y - Atoms.dot(u)
	magma_saxpy(N, -1.0 * u_temp, atom_d, 1, res_d, 1, queue);

	B_row_ptr = B;
	B_col_ptr = B;
	for (int i = 1; i < k; i++){
//		std::cout<<i<<'\n';
		magma_sgemv(MagmaTrans, N, M, alpha, A_d, N, res_d, incx, beta, dot_p, incx, queue);
		max_index = magma_isamax(M, dot_p, incx, queue) - 1;
		indx[i] = max_index;

		// Update atom
		A_ptr = A_d;
		A_ptr += max_index * N;
		magma_scopyvector(N, A_ptr, 1, atom_ptr, 1, queue);
		float d1 = magma_snrm2(N, A_ptr, 1, queue);

		magma_scopyvector(N, b_d, 1, res_d, 1, queue);

		atom_ptr += N;
//		magma_sprint_gpu(N, i+1, atom_d, N, queue);
//		magma_sprint_gpu(N, 1, A_ptr, N, queue);

		// u1
		magma_sgemv(MagmaTrans, N, i, 1.0, atom_d, N, A_ptr, 1, 0.0, u1, 1, queue);
//		magma_sprint_gpu(N, i+1, atom_d, N, queue);
		magma_sgemv(MagmaTrans, N, i+1, 1.0, atom_d, N, b_d, 1, 0.0, temp, 1, queue);

		// u2. ldda of B is always k
		magma_sgemv(MagmaNoTrans, i, i, 1.0, B, k, u1, 1, 0.0, u2, 1, queue);
		float d2 = magma_sdot(i, u1, 1, u2, 1, queue);
		float d = 1.0 / (d1 - d2);
		float *d_ptr = &d;
		// F
//		std::cout<<"here\n";

		magma_sgemm(MagmaNoTrans, MagmaTrans, i, i, 1, d, u2, i, u2, i, 1.0, B, k, queue);
//		magma_sprint_gpu(i, i, B, k, queue);

//		std::cout<<"here2\n";

		magma_sscal(i, -1.0 * d, u2, 1, queue);

		B_row_ptr += 1;
		B_col_ptr += k;

//		std::cout << B_row_ptr << ' ' << B_col_ptr << '\n';
//		magma_sprint_gpu(i, i, B, k, queue);

		magma_scopyvector(i, u2, 1, B_row_ptr, k, queue);
		magma_scopyvector(i, u2, 1, B_col_ptr, 1, queue);

		magma_ssetvector(1, d_ptr, 1, B_col_ptr + i, 1, queue);

//		std::cout<<d<<'\n';
//		magma_sprint_gpu(i+1, i+1, B, k, queue);


		// u
		magma_sgemv(MagmaNoTrans, i+1, i+1, 1.0, B, k, temp, 1, 0.0, u, 1, queue);


		// update res_d
		magma_sgemv(MagmaNoTrans, N, i+1, -1.0, atom_d, N, u, 1, 1.0, res_d, 1, queue);

//		magma_sprint_gpu(N,1, res_d, N, queue);
//		magma_sprint_gpu(N, i+1, atom_d, N, queue);

	}

//	indx = indx_d;

	magma_sgetvector(k, u, 1, x_T_h, 1, queue);


# pragma omp parallel for
	for (int i = 0; i < k; i++) {
		coeff[indx[i]] = x_T_h[i];
	}
	stop = second();

	std::cout<<"elapsed time: "<< stop-start << '\n';
	std::cout << "return sucessfully!\n";
}

void test(){
	magma_queue_t queue = NULL;
	magma_int_t dev = 0;
	magma_queue_create(dev, &queue);

	int N = 15;

	magmaFloat_ptr A;
	magma_smalloc(&A, N);
	std::cout<<"hi!\n";

	for (int i = 0; i < N; i++){
		*(A+i) = (float)i;
	}
	std::cout<<"hi!\n";

	magma_sprint_gpu(N, 1, A, N, queue);
	magma_free(A);
}
