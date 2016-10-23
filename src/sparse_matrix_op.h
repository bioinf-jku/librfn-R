/* This is the interface for RFN's sparse matrix operations. 
 * If you want to use the generic implementation, compile nist_spblas.cc, 
 * If you want to use the MKL, compile mkl_sparse_impl.cpp and link to MKL. */

typedef int spmat_t;

/* memory management */
spmat_t create(int row, int col); /* empty */
spmat_t suscr_csr(int m, int n, float *val, int *col, int *ptr); /* from csr */
void destroy(spmat_t A);

/* select row subset */
spmat_t srowsubset(spmat_t A, int first_row, int nrow); /* allocates new matrix */

/* column means and variances */
void scolmeans(spmat_t A, float *means);
void scolvars(spmat_t A, float *vars);

/* scale rows/cols */
void sscalecols(spmat_t A, float *s);
void sscalerows(spmat_t A, float *s);

/* set element (set to zero will delete entry) */
void ssetelement( spmat_t A, int row, int col, float val );
void ssetelement( spmat_t A, int idx, float val );

/* get element reference */
float &sgetelement( spmat_t A, int row, int col);
float &sgetelement( spmat_t A, int idx );

/* get element pointer */
float *sgetelementp( spmat_t A, int row, int col );
float *sgetelementp( spmat_t A, int idx );

/* sgemm routines with sparse matrix being lhs (A) or rhs (B) of the product */
void susgemm(char sidea, char transa, char transb, int nohs, const float &alpha, spmat_t A, 
   const float *B, int ldB, const float &beta, float *C, int ldC);

/* checks whether A is a valid handle */
bool handle_valid(spmat_t A);

/* debug */
namespace NIST_SPBLAS
{void print(int A);}

#include <cstdio>

#define dbg_printf(...) do { \
   printf(__VA_ARGS__); \
   fflush(stdout); \
} while (0)

