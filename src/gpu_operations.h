/*
 Copyright © 2015 Thomas Unterthiner
 Licensed under GPL, version 2 or a later (see LICENSE.txt)
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstring>
#include <ctype.h>
#include <cassert>
#include <map>
#include <cusparse_v2.h>
#include <typeinfo> /* for typeid */
#include "sparse_matrix.h"

using std::fprintf;

inline cublasFillMode_t uplo_to_cublas(const char* uplo) {
	return tolower(uplo[0]) == 'l' ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
}

static const char* cusparseErrorString(cusparseStatus_t error) {
	switch (error) {
	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";
	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";
	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";
	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	default:
		return "<unknown>";
	}
}

static const char* cublasErrorString(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
	default:
		return "<unknown>";
	}
}

#ifndef DNDEBUG

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#define CUBLAS_CALL(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line) {
//printf("%d (%s:%d)\n", code, file, line);
	if (code != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLAS Error: %s %s:%d\n", cublasErrorString(code), file, line);
		exit(code);
	}
}

#define CUSPARSE_CALL(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line) {
	// printf("%d (%s:%d)\n", code, file, line);
	if (code != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "CUSPARSE Error: %s %s:%d\n", cusparseErrorString(code), file, line);
		exit(code);
	}
}

static const char* cusolverErrorString(cusolverStatus_t error) {
	switch (error) {
	case CUSOLVER_STATUS_SUCCESS:
		return "CUSOLVER_STATUS_SUCCESS";
	case CUSOLVER_STATUS_NOT_INITIALIZED:
		return "CUSOLVER_STATUS_NOT_INITIALIZED";
	case CUSOLVER_STATUS_ALLOC_FAILED:
		return "CUSOLVER_STATUS_ALLOC_FAILED";
	case CUSOLVER_STATUS_INVALID_VALUE:
		return "CUSOLVER_STATUS_INVALID_VALUE";
	case CUSOLVER_STATUS_ARCH_MISMATCH:
		return "CUSOLVER_STATUS_ARCH_MISMATCH";
	case CUSOLVER_STATUS_EXECUTION_FAILED:
		return "CUSOLVER_STATUS_EXECUTION_FAILED";
	case CUSOLVER_STATUS_INTERNAL_ERROR:
		return "CUSOLVER_STATUS_INTERNAL_ERROR";
	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	default:
		return "<unknown>";
	}
}

#define CUSOLVER_CALL(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
	//printf("%d (%s:%d)\n", code, file, line);
	if (code != CUSOLVER_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLAS Error: %s %s:%d\n", cusolverErrorString(code), file, line);
		exit(code);
	}
}

#else
#define CUBLAS_CALL(ans) (ans)
#define CUDA_CALL(ans) (ans)
#define CUSOLVER_CALL(ans) (ans)
#define CUSPARSE_CALL(ans) (ans)
#endif

class GPU_Operations {
	cublasHandle_t handle;
	curandState* rng_state;
	cusolverDnHandle_t cudense_handle;
	cusparseHandle_t cusparse_handle;
	std::map<int, float*> buffer_map; // keeps track of buffers allocated for potrf
	int* devinfo; // cuSOLVER error reporting
public:

	float* ones;
	GPU_Operations(int n, int m, int k, unsigned long seed, int gpu_id);
	~GPU_Operations();

	float* to_device(const float* src, size_t size) const;
	unsigned* to_device(const unsigned* src, size_t size) const;
	sparseMatrix* to_device(const sparseMatrix* src, size_t size) const;

	float* to_host(float* src, float* dst, size_t size) const {
		CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
		free(src);
		return dst;
	}

	unsigned* to_host(unsigned* src, unsigned* dst, size_t size) const {
		CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
		free(src);
		return dst;
	}

	float* copy_to_host(const float* src, float* dst, size_t size) const {
		CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
		return dst;
	}

	unsigned* copy_to_host(const unsigned* src, unsigned* dst, size_t size) const {
		CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
		return dst;
	}

	void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
			const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
			const int ldc) const {
		cublasOperation_t ta = tolower(transa[0]) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t tb = tolower(transb[0]) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CALL(cublasSgemm(handle, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
	}

#define char_trans_to_cusparse(tr) (tr[0] == 'T' || tr[0] == 't' ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE)
#define char_trans_to_cusparse_rev(tr) (tr[0] == 'T' || tr[0] == 't' ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE)

	void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
			const sparseMatrix* a, const int lda, const float *b, const int ldb, const float beta, float *c,
			const int ldc) const {
		cusparseMatDescr_t descr;
		CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
		CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

		// Difference in API
		// 		cusparse: m number of rows of sparse matrix     A
		//		cublas:   m number of rows of        matrix op( A )
		// same for k
		cusparseOperation_t opA = char_trans_to_cusparse(transa);
		cusparseOperation_t opB = char_trans_to_cusparse(transb);
		unsigned m_a = m;
		unsigned n_a = k;
		if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE) {
			m_a = k;
			n_a = m;
		}

		CUSPARSE_CALL(cusparseScsrmm2(cusparse_handle, opA, opB, m_a, n, k_a,
				a->nnz, &alpha, descr, a->values, a->rowPointers, a->columns, b, ldb, &beta, c, ldc));
		CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
	}

	void gemm(const char *transa, const char *transb, const int m, const int n,
			const int k, const float alpha, const float *a, const int lda,
			const sparseMatrix* b, const int ldb, const float beta, float *c,
			const int ldc) const {
		// instead of A*B we have to calculate Bt*At and then transpose back the result
		cusparseMatDescr_t descr;
		CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
		CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

		cusparseOperation_t opA = char_trans_to_cusparse_rev(transa);
		cusparseOperation_t opB = char_trans_to_cusparse_rev(transb);
		unsigned nrow_b = k;
		unsigned ncol_b = n;
		if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE) {
			nrow_b = n;
			ncol_b = k;
		}

		float* c_t = malloc(nrow_b * ncol * sizeof(float));
		CUSPARSE_CALL(
				cusparseScsrmm2(cusparse_handle, opB, opA, nrow_b, k, ncol_b, b->nnz, &alpha, descr, b->values, b->rowPointers, b->columns, a, lda, &beta, c_t, ldc));

		float const al(1.0);
		float const be(0.0);

		CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nrow_b, k, &al, c_t, ldc, &be, 0, 0, c, ldc));

		CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
		free(c_t);
	}
#undef char_trans_to_cusparse
#undef char_trans_to_cusparse_rev

	void dgmm(const char* mode, const int m, const int n, const float* A, int lda, const float* x, int incx, float* C,
			int ldc) const {
		cublasSideMode_t lr = mode[0] == 'l' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
		CUBLAS_CALL(cublasSdgmm(handle, lr, m, n, A, lda, x, incx, C, ldc));
	}

	void symm(const char *side, const char *uplo, const int m, const int n, const float alpha, const float *a,
			const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc) const {
		cublasSideMode_t s = tolower(side[0]) == 'l' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
		cublasFillMode_t ul = uplo_to_cublas(uplo);
		CUBLAS_CALL(cublasSsymm(handle, s, ul, m, n, &alpha,a, lda, b, ldb, &beta, c, ldc));
	}

	void axpy(const int n, const float alpha, const float* x, const int incx, float *y, const int incy) const {
		CUBLAS_CALL(cublasSaxpy(handle, n, &alpha, x, incx, y, incy));
	}

	int potrf(const char *uplo, int n, float* a, int lda) {
		cublasFillMode_t ul = uplo_to_cublas(uplo);
		int bufsize = 0;
		int info = 0;
		CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(cudense_handle, ul, n, a, lda, &bufsize));

		// See if we already have a buffer of correct size, otherwise allocate
		float* buffer = 0;
		auto it = buffer_map.find(bufsize);
		if (it != buffer_map.end()) {
			buffer = it->second;
		} else {
			buffer = malloc(bufsize * sizeof(float));
			buffer_map[bufsize] = buffer;
		}

		CUSOLVER_CALL(cusolverDnSpotrf(cudense_handle, ul, n, a, lda, buffer, bufsize, devinfo));
		CUDA_CALL(cudaMemcpy(&info, devinfo, sizeof(info), cudaMemcpyDeviceToHost));
		return info;
	}

	int potrs(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) const {
		int info;
		cublasFillMode_t ul = uplo_to_cublas(uplo);
		CUSOLVER_CALL(cusolverDnSpotrs(cudense_handle, ul, n, nrhs, a, lda, b, ldb, devinfo));
		CUDA_CALL(cudaMemcpy(&info, devinfo, sizeof(info), cudaMemcpyDeviceToHost));
		return info;
	}

	int posv(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) {
		int info = potrf(uplo, n, a, lda);
		if (info == 0)
			info = potrs(uplo, n, nrhs, a, lda, b, ldb);
		return info;
	}

	void* memset(void* dest, int ch, size_t count) const {
		CUDA_CALL(cudaMemset(dest, ch, count));
		return dest;
	}

	float* memcpy(void* dest, const void *src, size_t count) const {
		CUDA_CALL(cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice));
		return 0;
	}

	void free(void* ptr) const {
		if (ptr != 0)
			CUDA_CALL(cudaFree(ptr));
	}

	void free_devicememory(void* ptr) const {
		if (ptr != 0)
			CUDA_CALL(cudaFree(ptr));
	}

	float* malloc(size_t size) const {
		float* retval = 0;
		cudaError_t err = cudaMalloc(&retval, size);
		CUDA_CALL(err);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed\n");
			retval = 0;
		}
		return retval;
	}

	template<typename T> T* malloc_t(size_t size) const {
		T* retval = 0;
		cudaError_t err = cudaMalloc(&retval, size);
		CUDA_CALL(err);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed\n");
			retval = 0;
		}
		return retval;
	}

	void fill_eye(float* X, unsigned n) const;
	void fill(float* X, const unsigned size, const float value) const;
	void maximum(float* x, const float value, const unsigned size) const;
	void leaky_relu(float* x, const float value, const unsigned size) const;
	void tanh(float* x, const unsigned size) const;
	void sigmoid(float* x, const unsigned size) const;
	void soft_threshold(float* x, const float alpha, const unsigned size) const;
	void invsqrt(float* s, const unsigned n) const;

	void invert(float* X, const unsigned size) const;

	void calculate_column_variance(const float* X, const unsigned nrows, const unsigned ncols, float* variances) const;
	void scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void dropout(float* X, const unsigned size, const float dropout_rate) const;
	void add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const;
	void add_gauss_noise(float* X, const unsigned size, const float noise_rate) const;

	void calculate_column_variance(const sparseMatrix* X, const unsigned nrows, const unsigned ncols, float* variances) const;
	void scale_columns(sparseMatrix* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(sparseMatrix* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void dropout(sparseMatrix* X, const unsigned size, const float dropout_rate) const;
	void add_saltpepper_noise(sparseMatrix* X, const unsigned size, const float noise_rate) const;
	void add_gauss_noise(sparseMatrix* X, const unsigned size, const float noise_rate) const;

// Useful for debugging
	void printMatrixCM(const float* a, int n, int m, const char* fmt);
	void printMatrixRM(const float* a, int n, int m, const char* fmt);

	void printMatrixSP(const sparseMatrix* a, const char* fmt) const;

	template<typename T>
	T init_invalid(void) {
		return (typeid(T) == typeid(sparseMatrix*) ? (T) -1 : (T) 0);
	}

	template<typename T>
	T malloc_matrix(int rows, int cols) {
		return malloc_matrix(rows, cols, init_invalid<T>());
	}

	sparseMatrix* malloc_matrix(int rows, int cols, sparseMatrix* dummy) {
		sparseMatrix* matrix = (sparseMatrix*) std::malloc(sizeof(sparseMatrix));
		return matrix;
	}

	float *malloc_matrix(int rows, int cols, float *dummy) {
		return malloc(rows * cols * sizeof(float));
	}

	float *memcpy_matrix(float *dest, float *src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
		return memcpy(dest, &src[first_row * src_ncol], nrows_to_copy * src_ncol * sizeof(float));
	}

	sparseMatrix* memcpy_matrix(sparseMatrix* dest, sparseMatrix* src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
		unsigned fromIndex = 0;
		unsigned toIndex   = 0;
		CUDA_CALL(cudaMemcpy(&fromIndex, src->rowPointers + first_row,                 sizeof(unsigned), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&toIndex  , src->rowPointers + first_row + nrows_to_copy, sizeof(unsigned), cudaMemcpyDeviceToHost));

		dest->nnz = (toIndex - fromIndex);
		dest->m = nrows_to_copy;

		dest->values      = malloc(dest->nnz * sizeof(float));
		dest->columns     = malloc_t<unsigned>(dest->nnz * sizeof(unsigned));
		dest->rowPointers = malloc_t<unsigned>((nrows_to_copy + 1) * sizeof(unsigned));

		memcpy(dest->values     , src->values      + fromIndex, dest->nnz * sizeof(float));
		memcpy(dest->columns    , src->columns     + fromIndex, dest->nnz * sizeof(unsigned));
		memcpy(dest->rowPointers, src->rowPointers + first_row, (nrows_to_copy + 1) * sizeof(unsigned));
		subtract_first_element(dest->rowPointers, nrows_to_copy + 1);

		return dest;
	}

	void prints(float* f, unsigned l) const {
			float* src = (float*) std::malloc(l * sizeof(float));
			copy_to_host(f, src, l * sizeof(float));
			for (unsigned i = 0; i < l; ++i) {
				printf("%f ", src[i]);
			}
			printf("\n");
			std::free(src);
		}

	void printsu(unsigned* f, unsigned l) const {
		unsigned* src = (unsigned*) std::malloc(l * sizeof(unsigned));
			copy_to_host(f, src, l * sizeof(unsigned));
			for (unsigned i = 0; i < l; ++i) {
				printf("%d ", src[i]);
			}
			printf("\n");
			std::free(src);
		}

	void subtract_first_element(unsigned* a, unsigned len) const;

	void free_sparse(void *ptr) {
	}

	void free_sparse(sparseMatrix* a) {
		if (handle_valid(a)) {
			free(a->columns);
			free(a->rowPointers);
			free(a->values);
		}
	}

	bool handle_valid(sparseMatrix* a) {
		return a->values != (float*)-1;
	}

	float* get_batch(const float* X, int ncol, int batch_num, int batch_size) {
		/* return pointer */
		return (float*) &X[batch_num * batch_size * ncol];
	}

	sparseMatrix* get_batch(sparseMatrix* X, int ldx, int batch_num, int batch_size) {
		sparseMatrix* dest = (sparseMatrix*) std::malloc(sizeof(sparseMatrix));
		memcpy_matrix(dest, X, batch_size, batch_num * batch_size);
		return dest;
	}

};
