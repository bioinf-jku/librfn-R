#include <Rcpp.h>
#include <vector>

#include "librfn.h"
#include "sparse_matrix_op.h"
#include "sparse_matrix.h"

#include <cstdio>
#include "use_R_impl.h"

#include <ctime>

using namespace Rcpp;

RcppExport SEXP train_rfn_cpu(SEXP Xs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP n_iters, 
   SEXP batch_sizes, SEXP etaWs, SEXP etaPs, SEXP minPs, SEXP h_thresholds, SEXP dropout_rates,
   SEXP input_noise_rates, SEXP l2_weightdecays, SEXP l1_weightdecays, SEXP momentums, 
   SEXP noise_types, SEXP apply_relus, SEXP apply_scalings, SEXP apply_newton_updates, SEXP seeds) 
{
   BEGIN_RCPP
   
   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);
   
   GetRNGstate();
   
   clock_t t = clock();
   train_cpu(&X[0], &W[0], &P[0], n, m, k, as<int>(n_iters), as<int>(batch_sizes), as<float>(etaWs),
      as<float>(etaPs), as<float>(minPs), as<float>(h_thresholds), as<float>(dropout_rates), 
      as<float>(input_noise_rates), as<float>(l2_weightdecays), as<float>(l1_weightdecays),
      as<float>(momentums), as<int>(noise_types), as<int>(apply_relus), as<int>(apply_scalings),
      as<int>(apply_newton_updates), as<int>(seeds));
   t = clock() - t;
   
   PutRNGstate();

   NumericVector W_ret = wrap(W);
   W_ret.attr("dim") = Dimension(m, k); /* conversion from rowmajor 2 colmajor */

   NumericVector P_ret = wrap(P);

   List ret;
   ret["W"] = W_ret;
   ret["P"] = P_ret;
   ret["T"] = wrap<double>(((double) t) / CLOCKS_PER_SEC);
   return ret;
   
   END_RCPP
}

RcppExport SEXP train_rfn_cpu_sparse(SEXP Xs, SEXP rowvs, SEXP colvs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP n_iters, 
   SEXP batch_sizes, SEXP etaWs, SEXP etaPs, SEXP minPs, SEXP h_thresholds, SEXP dropout_rates,
   SEXP input_noise_rates, SEXP l2_weightdecays, SEXP l1_weightdecays, SEXP momentums, 
   SEXP noise_types, SEXP apply_relus, SEXP apply_scalings, SEXP apply_newton_updates, SEXP seeds) 
{
   BEGIN_RCPP
   
   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);
   
   std::vector<int> rowv = as<std::vector<int> >(rowvs);
   std::vector<int> colv = as<std::vector<int> >(colvs);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);
   
   GetRNGstate();
   
   spmat_t X_sparse = suscr_csr(n, m, &X[0], &colv[0], &rowv[0]);

   clock_t t = clock();
   train_cpu_sparse(X_sparse, &W[0], &P[0], n, m, k, as<int>(n_iters), as<int>(batch_sizes), as<float>(etaWs),
      as<float>(etaPs), as<float>(minPs), as<float>(h_thresholds), as<float>(dropout_rates), 
      as<float>(input_noise_rates), as<float>(l2_weightdecays), as<float>(l1_weightdecays),
      as<float>(momentums), as<int>(noise_types), as<int>(apply_relus), as<int>(apply_scalings),
      as<int>(apply_newton_updates), as<int>(seeds));
   t = clock() - t;
   
   PutRNGstate();

   NumericVector W_ret = wrap(W);
   W_ret.attr("dim") = Dimension(m, k);

   NumericVector P_ret = wrap(P);

   List ret;
   ret["W"] = W_ret;
   ret["P"] = P_ret;
   ret["T"] = wrap<double>(((double) t) / CLOCKS_PER_SEC);
   return ret;
   
   END_RCPP
}

RcppExport SEXP train_rfn_gpu_sparse(SEXP Xs, SEXP rowvs, SEXP colvs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP n_iters,
   SEXP batch_sizes, SEXP etaWs, SEXP etaPs, SEXP minPs, SEXP h_thresholds, SEXP dropout_rates,
   SEXP input_noise_rates, SEXP l2_weightdecays, SEXP l1_weightdecays, SEXP momentums,
   SEXP noise_types, SEXP apply_relus, SEXP apply_scalings, SEXP apply_newton_updates, SEXP seeds)
{
   BEGIN_RCPP

   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);

   std::vector<int> rowv = as<std::vector<int> >(rowvs);
   std::vector<int> colv = as<std::vector<int> >(colvs);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);

   GetRNGstate();

   sparseMatrix sparse;
   sparse.m = n;
   sparse.nnz = rowv[n];
   sparse.values = &X[0];
   sparse.columns = &colv[0];
   sparse.rowPointers = &rowv[0];

   clock_t t = clock();
   train_gpu_sparse(X_sparse, &W[0], &P[0], n, m, k, as<int>(n_iters), as<int>(batch_sizes), as<float>(etaWs),
      as<float>(etaPs), as<float>(minPs), as<float>(h_thresholds), as<float>(dropout_rates),
      as<float>(input_noise_rates), as<float>(l2_weightdecays), as<float>(l1_weightdecays),
      as<float>(momentums), as<int>(noise_types), as<int>(apply_relus), as<int>(apply_scalings),
      as<int>(apply_newton_updates), as<int>(seeds));
   t = clock() - t;

   PutRNGstate();

   NumericVector W_ret = wrap(W);
   W_ret.attr("dim") = Dimension(m, k);

   NumericVector P_ret = wrap(P);

   List ret;
   ret["W"] = W_ret;
   ret["P"] = P_ret;
   ret["T"] = wrap<double>(((double) t) / CLOCKS_PER_SEC);
   return ret;

   END_RCPP
}


RcppExport SEXP calculate_W(SEXP Xs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP activations, 
   SEXP apply_scalings, SEXP h_thresholds)
{
   BEGIN_RCPP
   
   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);
   std::vector<float> Wout(k*m);

   calculate_W_cpu(&X[0], &W[0], &P[0], &Wout[0], n, m, k, as<int>(activations),
      as<int>(apply_scalings), as<float>(h_thresholds));

   NumericVector Wout_ret = wrap(Wout);
   Wout_ret.attr("dim") = Dimension(m, k);
   
   return Wout_ret;
   
   END_RCPP
}

RcppExport SEXP calculate_W_sparse(SEXP Xs, SEXP rowvs, SEXP colvs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP activations, 
   SEXP apply_scalings, SEXP h_thresholds)
{
   BEGIN_RCPP
   
   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);
   
   
   std::vector<int> rowv = as<std::vector<int> >(rowvs);
   std::vector<int> colv = as<std::vector<int> >(colvs);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);
   std::vector<float> Wout(k*m);
   
   spmat_t X_sparse = suscr_csr(n, m, &X[0], &colv[0], &rowv[0]);
   
   calculate_W_cpu_sparse(X_sparse, &W[0], &P[0], &Wout[0], n, m, k, as<int>(activations),
      as<int>(apply_scalings), as<float>(h_thresholds));

   NumericVector Wout_ret = wrap(Wout);
   Wout_ret.attr("dim") = Dimension(m, k);
   
   return Wout_ret;
   
   END_RCPP
}

#include <mkl_spblas.h>
#include <mkl_trans.h>
using std::cerr;
using std::endl;

RcppExport SEXP somatcopy(SEXP s_arows, SEXP s_acols, SEXP s_A)
{
   BEGIN_RCPP
   
   char order = 'c';//as<char>(s_order);
   char trans = 't';//as<char>(s_trans);
   size_t arows = as<size_t>(s_arows);
   size_t acols = as<size_t>(s_acols);
   std::vector<float> A = as<std::vector<float> >(s_A);
   size_t lda = arows;
   std::vector<float> B(A.size());
   size_t ldb = acols;
   
   //mkl_somatcopy(order, trans, arows, acols, 1.f, &A[0], lda, &B[0], ldb);
   
   NumericVector B_ret = wrap(B);
   B_ret.attr("dim") = Dimension(acols, arows);
   
   return B_ret;
   
   END_RCPP
}

RcppExport SEXP _gemm(SEXP s_m, SEXP s_n, SEXP s_k, SEXP s_aval, SEXP s_acol, SEXP s_aptr, SEXP s_B)
{
   BEGIN_RCPP
   
   int m = as<int>(s_m);
   int n = as<int>(s_n);
   int k = as<int>(s_k);
   std::vector<float> aval = as<std::vector<float> >(s_aval);
   std::vector<int> acol = as<std::vector<int> >(s_acol);
   std::vector<int> aptr = as<std::vector<int> >(s_aptr);
   
   cerr << aval.size() << " " << acol.size() << " " << aptr.size() << " " << endl;
   
   spmat_t A = suscr_csr(m, n, &aval[0], &acol[0], &aptr[0]);
   std::vector<float> B = as<std::vector<float> >(s_B);
   std::vector<float> *C = new std::vector<float>(m*k);
   
   cerr << "B.size() " << B.size() << endl; 
   cerr << "C.size() " << C->size() << endl; 
   NIST_SPBLAS::print(A);
   
   susgemm('l', 'n', 'n', k, 1.f, A, &B[0], k, 0.f, &(*C)[0], k);
   
   /*cerr << "C is" << endl;
   
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < k; j++)
         cerr << (*C)[i * k + j] << ", ";
      cerr << endl;
   }*/
   
   NumericVector C_ret = wrap(*C);
   C_ret.attr("dim") = Dimension(m, k);
   
   cerr << "returning C" << endl;
   return C_ret;
   
   END_RCPP
}

RcppExport SEXP gemm(SEXP s_sidea, SEXP s_transa, SEXP s_transb, SEXP s_m, SEXP s_n, SEXP s_nohs, 
                     SEXP s_aval, SEXP s_acol, SEXP s_aptr, SEXP s_B)
{
   BEGIN_RCPP
   
   char sidea = as<char>(s_sidea);
   char transa = as<char>(s_transa);
   char transb = as<char>(s_transb);
   
   int m = as<int>(s_m);
   int n = as<int>(s_n);
   int nohs = as<int>(s_nohs);
   int cdim = ((sidea == 'l') == (transa == 't') ? n : m);
   //int ldb = (sidea == 'l' ? nohs : m);
   int ldc = (sidea == 'l' ? nohs : (transa == 't' ? m : n));
   
   int ldb = ((sidea == 'l') == (transb == 't') ? ((sidea == 'l') == (transa == 't') ? m : n) : nohs);
   
   std::vector<float> aval = as<std::vector<float> >(s_aval);
   std::vector<int> acol = as<std::vector<int> >(s_acol);
   std::vector<int> aptr = as<std::vector<int> >(s_aptr);
   spmat_t A = suscr_csr(m, n, &aval[0], &acol[0], &aptr[0]);
   
   std::vector<float> B = as<std::vector<float> >(s_B);
   std::vector<float> *C = new std::vector<float>(cdim * nohs);
   
   susgemm(sidea, transa, transb, nohs, 1.f, A, &B[0], ldb, 0.f, &(*C)[0], ldc);
   
   NumericVector C_ret = wrap(*C);
   C_ret.attr("dim") = (sidea == 'l' ? Dimension(cdim, nohs) : Dimension(nohs, cdim));
   
   cerr << "returning C" << endl;
   return C_ret;
   
   END_RCPP
}






