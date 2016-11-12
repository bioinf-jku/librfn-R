
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "../librfn.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// random in (0, 1]
static double rand_unif(void) {
    return (rand())/(RAND_MAX+1.0);
}

static double rand_max(int max) {
	return rand() % max;
}

int cmpfunc (const void * a, const void * b){
   return ( *(int*)a - *(int*)b );
}

/*
// generates random samples from a 0/1 Gaussian via Box-Mueller
static double rand_normal(void) {
    return sqrt(-2.0*log(rand_unif())) * cos(2.0*M_PI*rand_unif());
}
*/

float time_diff(struct timeval *t2, struct timeval *t1) {
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    return diff / 1000000.0f;
}

#define SPARSE


int main(int argc, char** argv) {
    int n = 50000;
    int m = 784;
    int k = 2048;
    int n_iter = 10;
    int gpu_id = -1;

    if (argc > 1)
        k = atoi(argv[1]);

    if (argc > 2)
        n_iter = atoi(argv[2]);

    if (argc > 3)
        m = atoi(argv[3]);

    if (argc > 4)
        gpu_id = atoi(argv[4]);

#ifdef SPARSE
    int nnz = 10000;
    float* X = (float*) malloc(nnz*sizeof(float));
    for (int i = 0; i < nnz; ++i) {
    	X[i] = 5.0f* rand_unif() - 0.5;
    }
    int* col = (int*) malloc(nnz*sizeof(int));
    for (int i = 0; i < nnz; ++i) {
    	col[i] = rand_max(m);
    }
    qsort(col, nnz, sizeof(int), cmpfunc);

    int* rowPointer = (int*) malloc((n + 1) * sizeof(int));
    rowPointer[0] = 0;
    rowPointer[n] = nnz;
    for (int i = 1; i < n; ++i) {
    	rowPointer[i] = rand_max(n);
    }
    qsort(rowPointer, n + 1, sizeof(int), cmpfunc);

    sparseMatrix sp;
    sp.values = X;
    sp.columns = col;
    sp.rowPointers = rowPointer;
    sp.m = n;
    sp.nnz = nnz;
#else
    float* X = (float*) malloc(n*m*sizeof(float));
    for (int i = 0; i < n*m; ++i) {
    	X[i] = 5.0f* rand_unif() - 0.5;
    }
#endif


    float* W = (float*) malloc(n*k*sizeof(float));
    float* P = (float*) malloc(m*sizeof(float));

    for (int i = 0; i < n*k; ++i)
        W[i] = rand_unif() - 0.5;

    struct timeval t0, t1;
    gettimeofday(&t0, 0);

#ifdef SPARSE
    train_gpu_sparse(&sp, W, P, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, gpu_id);
#else
    train_gpu(X, W, P, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, gpu_id);
#endif
    //train_cpu(X, W, P, n, m, k, n_iter, 0.1, 0.1, 1e-2, 0.0, 0.0, 32);
    gettimeofday(&t1, 0);
    printf("time for rfn: %3.4fs\n", time_diff(&t1, &t0));
    free(X);
    free(W);
    free(P);
    return 0;
}
