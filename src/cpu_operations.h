#include <cstring>
#include <cstdlib>
#include <cmath>

#ifndef COMPILE_FOR_R
#include <cassert>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <typeinfo> /* for typeid */

#include "sparse_matrix_op.h"

extern "C" {
extern void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
		const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc);

extern void ssymm_(const char *side, const char *uplo, const int *m, const int *n, const float *alpha, const float *a,
		const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc);

extern void saxpy_(const int *n, const float *alpha, const float *dx, const int *incx, float *dy, const int *incy);
extern int spotrf_(const char *uplo, int *n, float *a, int * lda, int *info);
extern int spotrs_(const char *uplo, int *n, int *nrhs, float * a, int *lda, float *b, int *ldb, int *info);
extern int sposv_(const char *uplo, int *n, int *nrhs, float * a, int *lda, float *b, int *ldb, int *info);
extern int spotri_(const char *uplo, int *n, float *a, int *lda, int *info);
}

using std::cos;
using std::log;
using std::sqrt;

#ifdef COMPILE_FOR_R
#include "use_R_impl.h"
#endif

#ifndef COMPILE_FOR_R
using std::rand;
using std::srand;

// random in (0, 1]
inline double rand_unif(void) {
	return (rand() + 1.0) / (RAND_MAX + 1.0);
}

// generates random samples from a 0/1 Gaussian via Box-Mueller
inline double rand_normal(void) {
	return sqrt(-2.0 * log(rand_unif())) * cos(2.0 * M_PI * rand_unif());
}
#endif

inline double rand_exp(double lambda) /* inversion sampling */
{
	return -log(1 - rand_unif()) / lambda;
}

class CPU_Operations {
	float* var_tmp;

public:

	float* ones;

	template<typename T>
	T init_invalid(void) {
		return (typeid(T) == typeid(spmat_t) ? (T) -1 : (T) 0);
	}

	CPU_Operations(const int m, const int n, const int k, unsigned long seed, int gpu_id);
	~CPU_Operations();

	float* to_device(const float* src, const int size) const {
		return (float*) src;
	}

	spmat_t to_device(spmat_t src, const int size) const {
		return src;
	}

	float* to_host(const float* src, float* dest, const int size) const {
		return dest;
	}

	float* copy_to_host(const float* src, float* dst, size_t size) const {
		memcpy(dst, src, size);
		return dst;
	}

	float* get_batch(const float* X, int ncol, int batch_num, int batch_size) {
		/* return pointer */
		return (float*) &X[batch_num * batch_size * ncol];
	}

	spmat_t get_batch(spmat_t X, int ldx, int batch_num, int batch_size) {
		/* return copy */
		return srowsubset(X, batch_num * batch_size, batch_size);
	}

	void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
			const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
			const int ldc) const {
		sgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
	}

#define to_spmat_trans(t) (t[0] == 'T' || t[0] == 't' ? blas_trans : blas_no_trans)

	void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
			const spmat_t a, const int lda, const float *b, const int ldb, const float beta, float *c,
			const int ldc) const {
		/* The gemm interface is understood as a column-major routine. The sparse implementation,
		 * however, is row-major, so we need to compute B^T * A^T = C^T instead of A * B = C. The
		 * transposition is implicitly performed by A, B and C being column-major. */
		susgemm('r', transa[0], transb[0], n, alpha, a, b, ldb, beta, c, ldc);
	}

	void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
			const float *a, const int lda, const spmat_t b, const int ldb, const float beta, float *c,
			const int ldc) const {
		susgemm('l', transb[0], transa[0], m, alpha, b, a, lda, beta, c, ldc);
	}

#undef to_spmat_trans

	void dgmm(const char* mode, const int m, const int n, const float* A, int lda, const float* x, int incx, float* C,
			int ldc) const;

	void symm(const char *side, const char *uplo, const int m, const int n, const float alpha, const float *a,
			const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc) const {
		ssymm_(side, uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
	}

	void axpy(const int n, const float alpha, const float *dx, const int incx, float *dy, const int incy) const {
		saxpy_(&n, &alpha, dx, &incx, dy, &incy);
	}

	int posv(const char* uplo, int n, int nrhs, float* a, int lda, float* b, int ldb) const {
		int info;
		int retval = sposv_(uplo, &n, &nrhs, a, &lda, b, &ldb, &info);

		if (info != 0)
			printf("info: %d\n", info);

		assert(!info);

		return retval;
	}

	int potrf(const char *uplo, int n, float* a, int lda) const {
		int info;
		int retval = spotrf_(uplo, &n, a, &lda, &info);
		assert(!info);
		return retval;
	}

	int potrs(const char *uplo, int n, int nrhs, float* a, int lda, float *b, int ldb, int *info) const {
		return spotrs_(uplo, &n, &nrhs, a, &lda, b, &ldb, info);
	}

	int potri(const char *uplo, int n, float *a, int lda) const {
		int info;
		int retval = spotri_(uplo, &n, a, &lda, &info);
		assert(!info);
		return retval;
	}

	void* memset(void* dest, int ch, size_t count) const {
		return std::memset(dest, ch, count);
	}

	float* memcpy(void* dest, const void *src, size_t count) const {
		return (float*) std::memcpy(dest, src, count);
	}

	float *memcpy_matrix(float *dest, float *src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
		return memcpy(dest, &src[first_row * src_ncol], nrows_to_copy * src_ncol * sizeof(float));
	}

	spmat_t memcpy_matrix(spmat_t &dest, spmat_t src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
		free(dest);
		return dest = srowsubset(src, first_row, nrows_to_copy);
	}

	void free(void* ptr) const {
		if (ptr != 0)
			std::free(ptr);
	}

	void free(spmat_t a) const {
		if (handle_valid(a))
			destroy(a);
	}

	void free_sparse(void *ptr) {
	}

	void free_sparse(spmat_t a) {
		if (handle_valid(a))
			destroy(a);
	}

	void free_devicememory(void* ptr) const {
		;
	}

	void free_devicememory(spmat_t X) const {
	}

	template<typename T>
	T malloc_matrix(int rows, int cols) {
		return malloc_matrix(rows, cols, init_invalid<T>());
	}

	spmat_t malloc_matrix(int rows, int cols, spmat_t dummy) {
		return create(rows, cols);
	}

	float *malloc_matrix(int rows, int cols, float *dummy) {
		return malloc(rows * cols * sizeof(float));
	}

	float* malloc(size_t size) const {
		return (float*) std::malloc(size);
	}

	void maximum(float* x, const float value, const int size) const {
		for (int i = 0; i < size; ++i)
			x[i] = fmaxf(x[i], value);
	}

	void leaky_relu(float* x, const float value, const int size) {
		for (int i = 0; i < size; ++i)
			x[i] = (x[i] < 0.0f) ? x[i] * value : x[i];
	}

	void sigmoid(float* x, const int size) const {
		for (int i = 0; i < size; ++i) {
			x[i] = 1 / (1 + expf(-x[i]));
		}
	}

	void tanh(float* x, const int size) const {
		for (int i = 0; i < size; ++i) {
			x[i] = tanhf(x[i]);
		}
	}

	void fill_eye(float* a, int n) const {
		memset(a, 0, n * n * sizeof(float));
		for (int i = 0; i < n; ++i)
			a[i * n + i] = 1.0f;
	}

	void fill(float* X, const int size, const float value) const {
		for (int i = 0; i < size; ++i) {
			X[i] = value;
		}
	}

	void calculate_column_variance(const float* X, const unsigned nrows, const unsigned ncols, float* variances);
	void calculate_column_variance(spmat_t X, const unsigned nrows, const unsigned ncols, float* variances) {
		memset(variances, 0, ncols * sizeof(float));
		scolvars(X, variances);
	}

	void invsqrt(float* s, const unsigned n) const;

	void scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_columns(spmat_t X, const unsigned nrows, const unsigned ncols, float* s) const {
		sscalecols(X, s);
	}

	void scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
	void scale_rows(spmat_t X, const unsigned nrows, const unsigned ncols, float* s) const {
		sscalerows(X, s);
	}

	void dropout(float* X, const unsigned size, const float dropout_rate) const {
		assert(0.0f <= dropout_rate && dropout_rate <= 1.0f);
		for (unsigned i = 0; i < size; ++i)
			X[i] = rand_unif() < dropout_rate ? 0.0f : X[i];
	}

	void dropout(spmat_t X, const unsigned size, const float dropout_rate) const {
		assert(0.0f <= dropout_rate && dropout_rate <= 1.0f);
		for (unsigned i = 0; i < size; ++i)
			/* TODO: write a routine sgetlement that leaves X const */
			if (rand_unif() < dropout_rate) {
				float *v = sgetelementp(X, i);

				if (v != NULL)
					*v = 0.f;
			}
	}

	void add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const {
		assert(0.0f <= noise_rate && noise_rate <= 1.0f);
		for (unsigned i = 0; i < size; ++i) {
			if (rand_unif() < noise_rate) {
				X[i] = rand_unif() < 0.5 ? 0.0f : 1.0f;
			}
		}
	}

	void add_saltpepper_noise(spmat_t X, const unsigned size, const float noise_rate) const {
		assert(0.0f <= noise_rate && noise_rate <= 1.0f);
		for (unsigned i = 0; i < size; ++i) {
			if (rand_unif() < noise_rate) {
				float *v = sgetelementp(X, i);

				if (v != NULL)
					*v = (rand_unif() < 0.5 ? 0.0f : 1.0f);
			}
		}
	}

	void add_gauss_noise(float* X, const unsigned size, const float noise_rate) const {
		assert(0.0 <= noise_rate);
		for (unsigned i = 0; i < size; ++i)
			X[i] += rand_normal() * noise_rate;
	}

	/* gauss noise makes no sense on sparse matrices */
	void add_gauss_noise(spmat_t X, const unsigned size, const float noise_rate) const {
		assert(0.0 <= noise_rate);
		for (unsigned i = 0; i < size; ++i) {
			float *v = sgetelementp(X, i);

			if (v != NULL)
				*v += rand_normal() * noise_rate;
		}
	}

	void invert(float* X, const unsigned size) const {
		for (unsigned i = 0; i < size; ++i)
			X[i] = 1.0f / X[i];
	}

	void soft_threshold(float* x, const float alpha, const int size) const {
		float f;
		for (int i = 0; i < size; ++i) {
			f = x[i];
			x[i] = f > 0 ? fmaxf(0., f - alpha) : fminf(0., f + alpha);
		}
	}

// Useful for debugging
	static void printMatrixCM(const float* a, int n, int m, const char* fmt);
	static void printMatrixCM(const spmat_t a, int n, int m, const char *fmt) {
		NIST_SPBLAS::print(a);
	}

	static void printMatrixRM(const float* a, int n, int m, const char* fmt);
	static void printMatrixRM(const spmat_t a, int n, int m, const char *fmt) {
		NIST_SPBLAS::print(a);
	}

	void prints(const float* f, unsigned l) const {}

	void printsu(const int* f, unsigned l) const {}
};
