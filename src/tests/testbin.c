
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
	float* dense = (float*) malloc(n * m * sizeof(float));
	memset(dense, 0, n * m * sizeof(float));
	for (unsigned i = 0; i < sparse->m; i++) {
		for (unsigned j = sparse->rowPointers[i]; j < sparse->rowPointers[i + 1]; j++) {
			dense[i + sparse->columns[j]*n] = sparse->values[j];
		}
	}
	return dense;
}

sparseMatrix* dense_to_sparse(float* dense, int n, int m) {
	int nnz = 0;
	int* rowPointers = (int*) malloc((n + 1) * sizeof(int));
	rowPointers[0] = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (dense[i*m+j] != 0) {
				nnz++;

			}
		}
		rowPointers[i + 1] = nnz;
	}

	float* values = (float*) malloc(nnz * sizeof(float));
	int* columns = (int*) malloc(nnz * sizeof(int));
	int ind = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (dense[i*m+j] != 0) {
				values[ind] = dense[i*m+j];
				columns[ind] = j;
				ind++;
			}
		}
	}

	sparseMatrix* sparse = (sparseMatrix*) malloc(sizeof(sparseMatrix));
	sparse->values = values;
	sparse->columns = columns;
	sparse->rowPointers = rowPointers;
	sparse->m = n;
	sparse->nnz = nnz;
	return sparse;
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
	const char* format = "%1.3f ";
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j)
			printf(format, x[i + j * n]);
		printf("\n");
	}
	printf("\n");
}

void printfl(float*x, int n) {
	printMat(x, 1, n);
}

void printi(int* x, int n) {
	const char* format = "%d ";
	for (int i = 0; i < n; ++i) {
		printf(format, x[i]);
	}
	printf("\n");
}

int main(int argc, char** argv) {
	srand(123);

    int n = 50000;
    int m = 784;
    int k = 2048;
    int n_iter = 10;
    int gpu_id = -1;
    int sparse = 0;

    if (argc > 1)
    	sparse = atoi(argv[1]);

    if (argc > 2)
        n_iter = atoi(argv[2]);

    if (argc > 3)
        m = atoi(argv[3]);

    if (argc > 4)
        gpu_id = atoi(argv[4]);

    float dropout = 0.6;
    float* X = (float*) malloc(n*m * sizeof(float));
    for (int i = 0; i < n*m; ++i) {
    	X[i] = 5.0f * rand_unif() - 0.5f;
    	if (rand_unif() < dropout) {
    		X[i] = 0;
    	}
    }

    sparseMatrix* sp = dense_to_sparse(X, n, m);

    //printf("Matrix\n");
    //printMat(X, n, m);
    //printf("Sparse\n");
    //printfl(sp->values, sp->nnz);
    //printi(sp->rowPointers, sp->m + 1);
    //printi(sp->columns, sp->nnz);


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

    float* W3 = (float*) malloc(m*k*sizeof(float));
    float* P3 = (float*) malloc(m*sizeof(float));
    memcpy(W3, W1, m*k*sizeof(float));
    memcpy(P3, P1, m*sizeof(float));

    struct timeval t0, t1;

    //gettimeofday(&t0, 0);
    //train_cpu(de, W, P, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32);
    //gettimeofday(&t1, 0);
    //printf("time for cpu rfn: %3.4fs\n", time_diff(&t1, &t0));

    if (sparse == 0) {
    	gettimeofday(&t0, 0);
    	int retval = train_gpu(X, W1, P1, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, gpu_id);
    	gettimeofday(&t1, 0);
    	printf("time for gpu rfn(%d): %3.4fs\n", retval, time_diff(&t1, &t0));

    	//printf("W\n");
    	//printMat(W1, m, k);
    }
    if (sparse == 1) {
    	gettimeofday(&t0, 0);
    	int retval = train_gpu_sparse(sp, W2, P2, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, gpu_id);
    	gettimeofday(&t1, 0);
    	printf("time for gpu sparse rfn: %3.4fs\n", time_diff(&t1, &t0));

    	//printf("W\n");
    	//printMat(W2, m, k);
    }
    if (sparse == 3) {
    	gettimeofday(&t0, 0);
    	int retval = train_cpu(X, W3, P3, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32);
    	gettimeofday(&t1, 0);
    	printf("time for cpu rfn: %3.4fs\n", time_diff(&t1, &t0));

    	//printf("W\n");
    	//printMat(W3, m, k);
    }
    free(X);
    free(sp->columns);
    free(sp->rowPointers);
    free(sp->values);
    free(sp);
    free(W1);
    free(P1);
    free(W2);
    free(P2);
    free(W3);
    free(P3);
    return 0;
}
