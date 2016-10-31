#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

struct sparseMatrix {
	float *values = -1;
	unsigned *columns = -1;
	unsigned *rowPointers = -1;
	unsigned m = -1; // number of rows
	unsigned nnz = -1; // number of nonzero elements
};

#endif /* SPARSE_MATRIX_H */
