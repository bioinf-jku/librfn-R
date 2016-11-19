#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

    int n = 4;
    int m = 6;

    float dropout = 0.6;
    float* X = (float*) malloc(n * m * sizeof(float));
    for (int i = 0; i < n * m; ++i) {
    	X[i] = 5.0f * rand_unif() - 0.5f;
    	if (rand_unif() < dropout) {
    		X[i] = 0;
    	}
    }

    sparseMatrix* sp = dense_to_sparse(X, n, m);

    printMat()


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
