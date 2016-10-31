#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

struct sparseMatrix {
	float *values = (float*)-1;
	unsigned *columns = (unsigned*)-1;
	unsigned *rowPointers = (unsigned*)-1;
	unsigned m = 0; // number of rows
	unsigned nnz = 0; // number of nonzero elements
};

#endif /* SPARSE_MATRIX_H */
