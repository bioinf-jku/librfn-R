#include "catch.hpp"
#include "../cpu_operations.h"
#include "../gpu_operations.h"
#include "../sparse_matrix.h"
#include <iostream>

#include <sys/time.h>
float time_diff(struct timeval *t2, struct timeval *t1) {
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	return diff / 1000000.0f;
}

using namespace std;

TEST_CASE( "to_host_and_to_device", "[gpu]" ) {
	GPU_Operations op(6, 6, 6, 0, -1);
	float X_h[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	float* X_d = op.to_device(X_h, sizeof(X_h));

	float* X2_h = (float*) malloc(sizeof(X_h));
	op.copy_to_host(X_d, X2_h, sizeof(X_h));
	for (size_t i = 0; i < sizeof(X_h) / sizeof(X_h[0]); ++i) {
		CHECK(X_h[i] == X2_h[i]);
	}
	free(X2_h);
	op.free(X_d);
}

TEST_CASE( "memcpy matrix sparse", "[gpu]" ) {
	/*
	 * 1   0   0
	 * 0   0   8
	 * 5   3   4
	 */
	unsigned n = 3;
	unsigned m = 3;
	unsigned k = 2;
	GPU_Operations gpu_op(m, n, k, 0, -1);

	sparseMatrix mat;

	float values[] = {1, 8, 5, 3, 4};
	int columns[] = {0, 2, 0, 1, 2};
	int pointerIndex[] = {0, 1, 2, 5};

	mat.values      = gpu_op.to_device(values      , 5 * sizeof(float));
	mat.columns     = gpu_op.to_device(columns     , 5 * sizeof(int));
	mat.rowPointers = gpu_op.to_device(pointerIndex, 4 * sizeof(int));
	mat.m = 3;
	mat.nnz = 5;

	sparseMatrix dest;

	gpu_op.memcpy_matrix(&dest, &mat, 2, 0, 1);

	float* dest_values            = (float*) malloc(4 * sizeof(float));
	int* dest_columnPointers = (int*) malloc(4 * sizeof(int));
	int* dest_rowPointers    = (int*) malloc(3 * sizeof(int));

	gpu_op.to_host(dest.values     , dest_values         , 4 * sizeof(float));
	gpu_op.to_host(dest.rowPointers, dest_rowPointers    , 3 * sizeof(int));
	gpu_op.to_host(dest.columns    , dest_columnPointers , 4 * sizeof(int));

	float exp_values[]            = {8, 5, 3, 4};
	int exp_columnPointers[] = {2, 0, 1, 2};
	int exp_rowPointers[]    = {0, 1, 4};

	for (unsigned i = 0; i < dest.m + 1; ++i) {
		CHECK(exp_rowPointers[i] == dest_rowPointers[i]);
	}
	for (unsigned i = 0; i < dest.nnz; ++i) {
		CHECK(exp_values[i] == dest_values[i]);
		CHECK(exp_columnPointers[i] == dest_columnPointers[i]);
	}
	free(dest_values);
	free(dest_columnPointers);
	free(dest_rowPointers);
}

template<class OP>
float* test_variance(OP& op, float* X, unsigned nrows, unsigned ncols, float* expected) {
	float* var = (float*) op.malloc(ncols * sizeof(X[0]));
	op.calculate_column_variance(X, nrows, ncols, var);
	float* res = (float*) malloc(ncols * sizeof(X[0]));
	op.copy_to_host(var, res, ncols * sizeof(var[0]));
	for (size_t i = 0; i < 3; ++i) {
		CHECK(res[i] == expected[i]);
	}
	free(res);
	return var;
}

TEST_CASE( "Calculate Variance", "[operations]" ) {
	GPU_Operations gpu_op(512, 512, 512, 0, -1);
	CPU_Operations cpu_op(512, 512, 512, 0, -1);
	float X_h[] = { 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	float expected[] = { 2.25, 4, 12.25 };
	float* res_h = test_variance(cpu_op, X_h, 2, 3, expected);
	cpu_op.free(res_h);
	float* X_d = gpu_op.to_device(X_h, sizeof(X_h));
	float* res_d = test_variance(gpu_op, X_d, 2, 3, expected);
	gpu_op.free(res_d);
	gpu_op.free(X_d);
}

sparseMatrix* create_sparse_matrix_d(const GPU_Operations &gpu_op, const float* x, const int* c,
		const int* p, unsigned m, unsigned nnz) {
	sparseMatrix *mat = (sparseMatrix*) std::malloc(sizeof(sparseMatrix));
	mat->values = gpu_op.to_device(x, nnz * sizeof(float));
	mat->columns = gpu_op.to_device(c, nnz * sizeof(int));
	mat->rowPointers = gpu_op.to_device(p, (m + 1) * sizeof(int));
	mat->nnz = nnz;
	mat->m = m;

	return mat;
}

void free_sparse_matrix_d(const GPU_Operations &gpu_op, sparseMatrix* matrix) {
	gpu_op.free(matrix->values);
	gpu_op.free(matrix->rowPointers);
	gpu_op.free(matrix->columns);
	std::free(matrix);
}

void test_sparse_variance(const GPU_Operations &gpu_op, const float* x, const int* c,
		const int* p, unsigned m, unsigned n, unsigned nnz, const float* expected) {
	sparseMatrix* mat = create_sparse_matrix_d(gpu_op, x, c, p, m, nnz);

	float* vars_d = gpu_op.malloc(n * sizeof(float));
	gpu_op.calculate_column_variance(mat, m, n, vars_d);
	float* vars_h = (float*) std::malloc(n * sizeof(float));
	gpu_op.to_host(vars_d, vars_h, n * sizeof(float));
	for (unsigned i = 0; i < n; i++) {
		CHECK(std::abs(expected[i] - vars_h[i]) < 1e-3);
	}

	free_sparse_matrix_d(gpu_op, mat);
}

TEST_CASE( "Calculate Variance sparse", "[operations]" ) {
	GPU_Operations gpu_op(512, 512, 512, 0, -1);
	float X_h[] = { 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	int column[] = {0, 1, 2, 0, 1, 2};
	int pointer[] = {0, 3, 6};
	float expected[] = { 2.25, 4, 12.25 };
	test_sparse_variance(gpu_op, X_h, column, pointer, 2, 3, 6, expected);

	float x2[] = {5.0, 1.0};
	int c2[] = {0, 1};
	int p2[] = {0, 0, 1, 1, 2, 2, 2};
	float e2[] = {3.472222, 0.13889, 0.0, 0.0, 0.0, 0.0, 0.0};
	test_sparse_variance(gpu_op, x2, c2, p2, 6, 7, 2, e2);
}

TEST_CASE( "Scale rows sparse [GPU]", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float x[] = { 5.0, 1.0, 3.0, -2.0 };
	int c[] = {0, 1, 2, 3};
	int p[] = {0, 0, 1, 2, 4, 4};

	float s[] = { 2.0, 3.0, 4.0, 5.0, 6.0 };
	float e[] = { 15.0, 4.0, 15.0, -10.0};
	sparseMatrix* mat = create_sparse_matrix_d(gpu_op, x, c, p, 5, 4);
	float* s_d = gpu_op.to_device(s, 5 * sizeof(float));
	gpu_op.scale_rows(mat, 5, 4, s_d);

	float* vals_h = (float*) std::malloc(4 * sizeof(float));
	gpu_op.copy_to_host(mat->values, vals_h, 4 * sizeof(float));

	for (unsigned i = 0; i < 4; i++) {
		CHECK(e[i] == vals_h[i]);
	}

	std::free(vals_h);
	gpu_op.free(s_d);
	free_sparse_matrix_d(gpu_op, mat);
}

TEST_CASE( "Scale columns sparse [GPU]", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float x[] = { 5.0, 1.0, 3.0, -2.0 };
	int c[] = {0, 1, 2, 3};
	int p[] = {0, 0, 1, 2, 4, 4};

	float s[] = { 3.0, 2.0, 4.0, 8.0 };
	float e[] = { 15.0, 2.0, 12.0, -16.0};
	sparseMatrix* mat = create_sparse_matrix_d(gpu_op, x, c, p, 5, 4);
	float* s_d = gpu_op.to_device(s, 4 * sizeof(float));
	gpu_op.scale_columns(mat, 5, 4, s_d);

	float* vals_h = (float*)std::malloc(4 * sizeof(float));
	gpu_op.copy_to_host(mat->values, vals_h, 4 * sizeof(float));

	for (unsigned i = 0; i < 4; i++) {
		CHECK(e[i] == vals_h[i]);
	}

	std::free(vals_h);
	gpu_op.free(s_d);
	free_sparse_matrix_d(gpu_op, mat);
}

TEST_CASE( "gemm sparse GPU", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float x[] = { 5.0, 1.0, 3.0, -2.0 };
	int c[] = {0, 1, 2, 3};
	int p[] = {0, 0, 1, 2, 4, 4};

	float b[] = { 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, -1.0 };
	float e[] = { 0.0, 5.0, 2.0, 1.0, 0.0, 0.0, 15.0, 2.0, 5.0, 0.0};
	sparseMatrix* mat = create_sparse_matrix_d(gpu_op, x, c, p, 5, 4);
	float* b_d = gpu_op.to_device(b, 8 * sizeof(float));
	float* c_d = gpu_op.malloc(2 * 5 * sizeof(float));

	int m = 5;
	int n = 2;
	int k = 4;

	gpu_op.gemm("n", "n", m, n, k, 1.0, mat, m, b_d, k, 0.0, c_d, m);

	float* c_h = (float*) std::malloc(2 * 5 * sizeof(float));

	gpu_op.to_host(c_d, c_h, 2 * 5 * sizeof(float));

	for (unsigned i = 0; i < 2 * 5; i++) {
		CHECK(c_h[i] == e[i]);
	}

	std::free(c_h);
	gpu_op.free(b_d);
	free_sparse_matrix_d(gpu_op, mat);
}

TEST_CASE( "gemm sparse GPU 2nd variant", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float x[] = { 5.0, 1.0, 3.0, -2.0 };
	int c[] = {0, 1, 2, 3};
	int p[] = {0, 0, 1, 2, 4, 4};

	float a[] = { 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 4.0, 1.0, 0.0, 3.0 };
	float e[] = { 10.0, 10.0, 3.0, 2.0, 12.0, 3.0, -8.0, -2.0 };


	int m = 2;
	int n = 4;
	int k = 5;

	sparseMatrix* b = create_sparse_matrix_d(gpu_op, x, c, p, k, 4);
	float* a_d = gpu_op.to_device(a, m * k * sizeof(float));
	float* c_d = gpu_op.malloc(m * n * sizeof(float));

	gpu_op.gemm("n", "n", m, n, k, 1.0, a_d, m, b, k, 0.0, c_d, m);

	float* c_h = (float*) std::malloc(m * n * sizeof(float));

	gpu_op.to_host(c_d, c_h, m * n * sizeof(float));

	for (unsigned i = 0; i < n * m; i++) {
		CHECK(c_h[i] == e[i]);
	}

	std::free(c_h);
	gpu_op.free(a_d);
	free_sparse_matrix_d(gpu_op, b);
}

TEST_CASE( "gemm sparse GPU 2nd variant transpose", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float x[] = { 5.0, 1.0, 3.0, -2.0 };
	int c[] = {0, 1, 2, 3};
	int p[] = {0, 0, 1, 2, 4, 4};

	float a[] = { 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 3.0, 4.0, 5.0, 0.0, 3.0 };
	float e[] = { 5.0, 0.0, 20.0, 2.0, 2.0, 5.0, 3.0, 0.0, 0.0, -2.0, 0.0, 0.0 };


	int m = 3;
	int n = 4;
	int k = 5;

	sparseMatrix* b = create_sparse_matrix_d(gpu_op, x, c, p, k, 4);
	float* a_d = gpu_op.to_device(a, m * k * sizeof(float));
	float* c_d = gpu_op.malloc(m * n * sizeof(float));

	gpu_op.gemm("t", "n", m, n, k, 1.0, a_d, k, b, k, 0.0, c_d, m);

	float* c_h = (float*) std::malloc(m * n * sizeof(float));

	gpu_op.to_host(c_d, c_h, m * n * sizeof(float));

	for (unsigned i = 0; i < n * m; i++) {
		CHECK(c_h[i] == e[i]);
	}

	std::free(c_h);
	gpu_op.free(a_d);
	free_sparse_matrix_d(gpu_op, b);
}

TEST_CASE( "gemm sparse GPU 2nd variant transpose and addition", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float x[] = { 5.0, 1.0, 3.0, -2.0 };
	int c[] = {0, 1, 2, 3};
	int p[] = {0, 0, 1, 2, 4, 4};

	float a[] = { 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 3.0, 4.0, 5.0, 0.0, 3.0 };
	float e[] = { 6.0, 1.0, 21.0, 4.0, 4.0, 7.0, 5.0, 2.0, 2.0, 0.0, 2.0, 2.0 };


	int m = 3;
	int n = 4;
	int k = 5;

	sparseMatrix* b = create_sparse_matrix_d(gpu_op, x, c, p, k, 4);
	float* a_d = gpu_op.to_device(a, m * k * sizeof(float));
	float* c_d = gpu_op.malloc(m * n * sizeof(float));
	gpu_op.fill(c_d, m, 2.0f);
	gpu_op.fill(&c_d[m], (n - 1) * m, 4.0f);

	gpu_op.gemm("t", "n", m, n, k, 1.0, a_d, k, b, k, 0.5, c_d, m);

	float* c_h = (float*) std::malloc(m * n * sizeof(float));

	gpu_op.to_host(c_d, c_h, m * n * sizeof(float));

	for (unsigned i = 0; i < n * m; i++) {
		CHECK(c_h[i] == e[i]);
	}

	std::free(c_h);
	gpu_op.free(a_d);
	free_sparse_matrix_d(gpu_op, b);
}

TEST_CASE( "get_batch sparse", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float x[] = { 5.0, 1.0, 3.0, -2.0 };
	int c[] = {0, 1, 2, 3};
	int p[] = {0, 0, 1, 2, 4, 4};

	float x_e[] = {1.0, 3.0, -2.0};
	int c_e[] = {1, 2, 3};
	int p_e[] = {0, 1, 3, 3};

	int m = 5;
	int n = 4;

	sparseMatrix* b = create_sparse_matrix_d(gpu_op, x, c, p, m, 4);

	sparseMatrix* b_batch = gpu_op.get_batch(b, 0, 1, 2);

	float* x_d = (float*) std::malloc(b_batch->nnz * sizeof(float));
	int* c_d = (int*) std::malloc(b_batch->nnz * sizeof(int));
	int* p_d = (int*) std::malloc((b_batch->m + 1) * sizeof(int));
	gpu_op.copy_to_host(b_batch->values, x_d, b_batch->nnz * sizeof(float));
	gpu_op.copy_to_host(b_batch->columns, c_d, b_batch->nnz * sizeof(int));
	gpu_op.copy_to_host(b_batch->rowPointers, p_d, (b_batch->m + 1)* sizeof(int));

	for (unsigned i = 0; i < b_batch->nnz; i++) {
		CHECK(x_d[i] == x_e[i]);
	}
	for (unsigned i = 0; i < b_batch->nnz; i++) {
		CHECK(c_d[i] == c_e[i]);
	}
	for (unsigned i = 0; i < b_batch->m + 1; i++) {
		CHECK(p_d[i] == p_e[i]);
	}

	std::free(x_d);
	std::free(c_d);
	std::free(p_d);
	//gpu_op.free(a_d);
	//free_sparse_matrix_d(gpu_op, b);
}

TEST_CASE( "gemm dense GPU", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float a[] = { 0, 5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, -2, 0};
	float b[] = { 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, -1.0 };

	float e[] = { 0.0, 5.0, 2.0, 1.0, 0.0, 0.0, 15.0, 2.0, 5.0, 0.0};

	int m = 5;
	int n = 2;
	int k = 4;

	float* a_d = gpu_op.to_device(a, m * k * sizeof(float));
	float* b_d = gpu_op.to_device(b, k * n * sizeof(float));
	float* c_d = gpu_op.malloc(n * m * sizeof(float));

	gpu_op.gemm("n", "n", m, n, k, 1.0, a_d, m, b_d, k, 0.0, c_d, m);

	float* c_h = (float*) std::malloc(m * m * sizeof(float));

	gpu_op.to_host(c_d, c_h, 2 * 5 * sizeof(float));

	for (unsigned i = 0; i < 2 * 5; i++) {
		CHECK(c_h[i] == e[i]);
	}

	std::free(c_h);
	gpu_op.free(b_d);
	gpu_op.free(a_d);
}

TEST_CASE( "gemm dense GPU 2", "[operations]" ) {
	GPU_Operations gpu_op(1, 1, 1, 0, -1);

	float a[] = { 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 4.0, 1.0, 0.0, 3.0 };
	float b[] = { 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, -2.0, 0.0, 0.0, 0.0, 0.0};

	float e[] = { 10.0, 10.0, 3.0, 2.0, 12.0, 3.0, -8.0, -2.0 };

	int m = 5;
	int n = 2;
	int k = 4;

	float* a_d = gpu_op.to_device(a, m * k * sizeof(float));
	float* b_d = gpu_op.to_device(b, k * n * sizeof(float));
	float* c_d = gpu_op.malloc(n * m * sizeof(float));

	gpu_op.gemm("n", "n", m, n, k, 1.0, a_d, m, b_d, k, 0.0, c_d, m);

	float* c_h = (float*) std::malloc(m * m * sizeof(float));

	gpu_op.to_host(c_d, c_h, 2 * 5 * sizeof(float));

	for (unsigned i = 0; i < 2 * 5; i++) {
		CHECK(c_h[i] == e[i]);
	}

	std::free(c_h);
	gpu_op.free(b_d);
	gpu_op.free(a_d);
}

// the pointer-to-memberfunction thingy is pretty ugly :(
template<class OP>
float* test_scale(OP& op, void (OP::*scalefunc)(float*, unsigned int, unsigned int, float*) const, float* X, float* s,
		unsigned nrows, unsigned ncols, float* expected) {
	float* scale = op.to_device(s, ncols * sizeof(X[0]));
	(op.*scalefunc)(X, nrows, ncols, scale);
	float* res = (float*) malloc(ncols * nrows * sizeof(X[0]));
	op.copy_to_host(X, res, ncols * nrows * sizeof(X[0]));
	for (size_t i = 0; i < nrows * ncols; ++i) {
		CHECK(expected[i] == res[i]);
	}
	free(res);
	return 0;
}

TEST_CASE( "Scale columns CPU", "[operations]" ) {
	CPU_Operations op(6, 6, 6, 0, -1);
	float X_h[] = { 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	float s_h[] = { 1.0, 2.0, 3.0 };
	float Exp_h[] = { 1.0, 4.0, 9.0, 4.0, 12.0, 30.0 };
	test_scale(op, &CPU_Operations::scale_columns, X_h, s_h, 2, 3, Exp_h);
}

TEST_CASE( "Scale columns GPU", "[operations]" ) {
	GPU_Operations op(6, 6, 6, 0, -1);
	float X_h[] = { 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	float s_h[] = { 1.0, 2.0, 3.0 };
	float Exp_h[] = { 1.0, 4.0, 9.0, 4.0, 12.0, 30.0 };
	float* X_d = op.to_device(X_h, sizeof(X_h));
	test_scale(op, &GPU_Operations::scale_columns, X_d, s_h, 2, 3, Exp_h);
	op.free(X_d);
}

TEST_CASE( "Scale rows CPU", "[operations]" ) {
	CPU_Operations op(6, 6, 6, 0, -1);
	float X_h[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 6.0, 10.0, 1.0, 1.5 };
	float s_h[] = { 2.0, 4.0 };
	float Exp_h[] = { 2.0, 4.0, 6.0, 8.0, 10.0, 16.0, 24.0, 40.0, 4.0, 6.0 };
	test_scale(op, &CPU_Operations::scale_rows, X_h, s_h, 2, 5, Exp_h);
}

TEST_CASE( "Scale rows GPU", "[operations]" ) {
	GPU_Operations op(6, 6, 6, 0, -1);
	float X_h[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 6.0, 10.0, 1.0, 1.5 };
	float s_h[] = { 2.0, 4.0 };
	float Exp_h[] = { 2.0, 4.0, 6.0, 8.0, 10.0, 16.0, 24.0, 40.0, 4.0, 6.0 };
	float* X_d = op.to_device(X_h, sizeof(X_h));
	test_scale(op, &GPU_Operations::scale_rows, X_d, s_h, 2, 5, Exp_h);
	op.free(X_d);
}

TEST_CASE( "invsqrt cpu", "[operations]" ) {
	CPU_Operations op(6, 6, 6, 0, -1);
	float x_h[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	float e_h[] = { 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	int n = sizeof(x_h) / sizeof(x_h[0]);
	for (int i = 1; i < n; ++i)
		e_h[i] = 1.0f / sqrt(x_h[i]);
	op.invsqrt(x_h, n);
	for (size_t i = 0; i < 3; ++i) {
		CHECK(abs(x_h[i] - e_h[i]) < 1e-3);
	}
}

TEST_CASE( "invsqrt gpu", "[operations]" ) {
	GPU_Operations op(6, 6, 6, 0, -1);
	float x_h[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	float e_h[] = { 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0 };
	int n = sizeof(x_h) / sizeof(x_h[0]);
	for (int i = 1; i < n; ++i)
		e_h[i] = 1.0f / sqrt(x_h[i]);
	float* x_d = op.to_device(x_h, sizeof(x_h));
	op.invsqrt(x_d, n);
	float* res = (float*) malloc(n * sizeof(x_h[0]));
	op.copy_to_host(x_d, res, n * sizeof(x_h[0]));
	for (size_t i = 0; i < 3; ++i) {
		CHECK(abs(res[i] - e_h[i]) < 1e-3);
	}
	op.free(x_d);
}

TEST_CASE( "filleye cpu", "[operations]" ) {
	unsigned n = 10;
	CPU_Operations op(n, n, n, 0, -1);
	float* x = op.malloc(n * n * sizeof(float));
	op.fill_eye(x, 10);
	double s = 0.0;
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			if (i == j) {
				CHECK(x[i * n + j] == 1.0);
			} else {
				s += abs(x[i * n + j]);
			}
		}
	}
	CHECK(s == 0.0);
	op.free(x);
}

TEST_CASE( "filleye gpu", "[operations]" ) {
	unsigned n = 10;
	CPU_Operations cpu_op(n, n, n, 0, -1);
	GPU_Operations op(n, n, n, 0, -1);
	float* x_d = op.malloc(n * n * sizeof(float));
	op.fill_eye(x_d, 10);
	float *x = cpu_op.malloc(n * n * sizeof(float));
	op.copy_to_host(x_d, x, n * n * sizeof(float));
	double s = 0.0;
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			if (i == j) {
				CHECK(x[i * n + j] == 1.0);
			} else {
				s += abs(x[i * n + j]);
			}
		}
	}
	CHECK(s == 0.0);
	op.free(x_d);
}

TEST_CASE( "Variance of CPU/GPU on large matrices", "[cpu_vs_gpu]" ) {
	unsigned n = 428;
	unsigned m = 554;
	CPU_Operations cpu_op(m, n, m, 0, -1);
	GPU_Operations gpu_op(m, n, m, 0, -1);

	float* X_h = cpu_op.malloc(n * m * sizeof(float));
	for (unsigned i = 0; i < n * m; ++i) {
		X_h[i] = 10 * ((rand() + 1.0) / (RAND_MAX + 1.0)) - 5.0;
	}
	float *X_d = gpu_op.to_device(X_h, n * m * sizeof(float));

	float* var_h = cpu_op.malloc(m * sizeof(float));
	float* var_d = gpu_op.malloc(m * sizeof(float));
	cpu_op.calculate_column_variance(X_h, n, m, var_h);
	gpu_op.calculate_column_variance(X_d, n, m, var_d);
	float* var_gpu_h = cpu_op.malloc(m * sizeof(float));
	gpu_op.to_host(var_d, var_gpu_h, m * sizeof(float));

	for (unsigned i = 0; i < m; ++i)
		CHECK(abs(var_h[i] - var_gpu_h[i]) < 1e-3);
	cpu_op.free(var_h);
	cpu_op.free(var_gpu_h);
}

TEST_CASE( "dgmm CPU/GPU", "[operations]" ) {
	unsigned n = 10;
	unsigned k = 10;
	unsigned m = 12;
	CPU_Operations cpu_op(m, n, k, 0, -1);
	GPU_Operations gpu_op(m, n, k, 0, -1);
	float* xh = cpu_op.malloc(m * k * sizeof(float));
	float* ah = cpu_op.malloc(m * sizeof(float));
	float* ch = cpu_op.malloc(m * k * sizeof(float));
	for (int i = 0; i < m * n; ++i)
		xh[i] = 10 * (rand() / RAND_MAX);
	for (int i = 0; i < n; ++i)
		ah[i] = 50 * (rand() / RAND_MAX);
	cpu_op.dgmm("l", m, k, xh, m, ah, 1, ch, m);

	float* xd = gpu_op.to_device(xh, m * k * sizeof(float));
	float* ad = gpu_op.to_device(ah, m * sizeof(float));
	float* cd = gpu_op.to_device(ch, m * k * sizeof(float));
	gpu_op.dgmm("l", m, k, xd, m, ad, 1, cd, m);

	float* dh = cpu_op.malloc(m * k * sizeof(float));
	gpu_op.copy_to_host(cd, dh, m * k * sizeof(float));
	for (unsigned i = 0; i < m * k; ++i) {
		CHECK(ch[i] == dh[i]);
	}
}
