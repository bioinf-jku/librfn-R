#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

struct sparseMatrix {
	float *values;
	unsigned *columns;
	unsigned *rowPointers;
	unsigned m; // number of rows
	unsigned nnz; // number of nonzero elements
};
