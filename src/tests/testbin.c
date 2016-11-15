
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

float* sparse_to_dense(sparseMatrix* sparse, int n, int m) {
	float* dense = (float*) malloc(n*m*sizeof(float));
	memset(dense, 0, n*m*sizeof(float));
	for (unsigned i = 0; i < sparse->m; i++) {
		for (unsigned j = sparse->rowPointers[i]; j < sparse->rowPointers[i + 1]; j++) {
			dense[i*m + sparse->columns[j]] = sparse->values[j];
		}
	}
	return dense;
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

void printMat(float* x, int n, int m) {
	char fmt = 0;
	const char* format = fmt == 0 ? "%1.3f " : fmt;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j)
				printf(format, x[i + j * n]);
			printf("\n");
		}
		printf("\n");
}

int main(int argc, char** argv) {
	srand(123);

    int n = 10;
    int m = 20;
    int k = 6;
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

    int nnz = 200;
    float* X = (float*) malloc(nnz*sizeof(float));
    for (int i = 0; i < nnz; ++i) {
    	X[i] = 5.0f* rand_unif() - 0.5;
    }

    int* rowPointer = (int*) malloc((n + 1) * sizeof(int));
    rowPointer[0] = 0;
    rowPointer[n] = nnz;
    for (int i = 1; i < n; ++i) {
    	rowPointer[i] = rand_max(nnz);
    }
    qsort(rowPointer, n + 1, sizeof(int), cmpfunc);

    int* col = (int*) malloc(nnz*sizeof(int));
    for (int i = 0; i < nnz; ++i) {
       col[i] = rand_max(m);
    }
    for (int i = 0; i < n; i++) {
    	qsort(&col[rowPointer[i]], rowPointer[i+1] - rowPointer[i], sizeof(int), cmpfunc);
    }

    sparseMatrix sp;
    sp.values = X;
    sp.columns = col;
    sp.rowPointers = rowPointer;
    sp.m = n;
    sp.nnz = nnz;

    float* de = sparse_to_dense(&sp, n, m);

    float* W1 = (float*) malloc(m*k*sizeof(float));
    float* P1 = (float*) malloc(m*sizeof(float));

    for (int i = 0; i < m*k; ++i)
       W1[i] = rand_unif() - 0.5;
    for (int i = 0; i < m; ++i)
       P1[i] = rand_unif() - 0.5;

    float* W2 = (float*) malloc(m*k*sizeof(float));
    float* P2 = (float*) malloc(m*sizeof(float));
    memcpy(W2, W1, m*k*sizeof(float));
    memcpy(P2, P1, m*sizeof(float));

    struct timeval t0, t1;

    //gettimeofday(&t0, 0);
    //train_cpu(de, W, P, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32);
    //gettimeofday(&t1, 0);
    //printf("time for cpu rfn: %3.4fs\n", time_diff(&t1, &t0));

    gettimeofday(&t0, 0);
    int retval = train_gpu(de, W1, P1, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, gpu_id);
    gettimeofday(&t1, 0);
    printf("time for gpu rfn(%d): %3.4fs\n", retval, time_diff(&t1, &t0));

    printf("W1\n");
    printMat(W1, m, k);

    gettimeofday(&t0, 0);
    train_gpu_sparse(&sp, W2, P2, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, gpu_id);
    gettimeofday(&t1, 0);
    printf("time for gpu sparse rfn: %3.4fs\n", time_diff(&t1, &t0));

    printf("W2\n");
    printMat(W2, m, k);

    free(X);
    free(col);
    free(rowPointer);
    free(de);
    free(W1);
    free(P1);
    free(W2);
    free(P2);
    return 0;
}
