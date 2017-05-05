/*
 * OMP_alt.h
 *
 *  Created on: Apr 3, 2017
 *      Author: kien
 */

#ifndef OMP_ALT_H_
#define OMP_ALT_H
#include <cuda.h>
#include <iostream>
#include "magma_v2.h"
#include <cublas_v2.h>
#include "magma_lapack.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <sys/time.h>

using namespace std;

void helloworld();
void OMP_alt(const float* A, const float* b, float* coeff, int k, int N, int M);
void test();
double second(void);

#endif /* OMP_ALT_H_ */
