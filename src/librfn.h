#ifndef LIBRFN_H
#define LIBRFN_H

#ifndef NOGPU
#include "sparse_matrix.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Trains an RFN network on the CPU.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory.
 *
 * @param X             [n, m] data matrix, with 1 sample per row
 * @param W             [k, m] weight matrix, expected to be pre-initialized
 * @param P             [m, ] vector, used to store Psi
 * @param n             number of samples
 * @param m             number of input features
 * @param k             number of hidden units
 * @param n_iter        number of iterations the algorithm will run
 * @param learnrate     learnrate
 * @param dropout_rate  the dropout rate for hidden activations
 * @param input_dropout_rate  the dropout rate for input units
 * @param seed          seed for the random number generation
 *
 * @return 0 on success, 1 otherwise. The trained network will be stored
 *         in the W and P variables.
 *
 */
int train_cpu(const float* X, float* W, float* P, const int n, const int m,
              const int k, const int n_iter, int batch_size, const float etaW,
              const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int noise_type, const int apply_relu, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed);

/* like train_cpu but X is a sparse matrix */
int train_cpu_sparse(int X, float* W, float* P, const int n, const int m,
              const int k, const int n_iter, int batch_size, const float etaW,
              const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int input_noise_type, const int activation_type, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed);


/**
 * Given a trained RFN, this will calculate the weights that are used to
 * estimate the hidden activations.
 *
 * This needs access to the training data, as the W need to incorporate
 * the scaling that would otherwise be done on the hidden activations.
 * The scaling parameters have to be fitted on the training data's H.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory.
 *
 * @param X             [n, m] training data matrix, with 1 sample per row
 * @param W             [k, m] RFN weight matrix
 * @param P             [m] vector, contains Psi
 * @param Wout          [k, m] output weight matrix
 * @param n             number of training samples
 * @param m             number of input features
 * @param k             number of hidden units
 */
void calculate_W_cpu(const float* X, const float* W, const float* P, float* Wout,
                     const int n, const int m, const int k,
                     const int activation_type, const int apply_scaling, const float h_threshold);
void calculate_W_cpu_sparse(int X, const float* W, const float* P, float* Wout,
                     const int n, const int m, const int k,
                     const int activation_type, const int apply_scaling, const float h_threshold);


#ifndef NOGPU
/**
 * Trains an RFN network on the GPU.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory. All transfers from and to the GPU will be
 * done internally by the function itself.
 *
 * @param X_host        [n, m] data matrix, with 1 sample per row
 * @param W_host        [k, m] weight matrix, expected to be pre-initialized
 * @param P_host        [m, ] vector, used to store Psi
 * @param n             number of samples
 * @param m             number of input features
 * @param k             number of hidden units
 * @param n_iter        number of iterations the algorithm will run
 * @param learnrate     learnrate
 * @param dropout_rate  the dropout rate for hidden activations
 * @param input_dropout_rate  the dropout rate for input units
 * @param seed          seed for the random number generation
 * @param gpu_id        ID of the GPU that this will run on (if this is -1,
 *                      the GPU with the most free memory will be picked)
 *
 * @return 0 on success, 1 otherwise. The trained network will be stored
 *         in the W_host and P_host variables.
 */
int train_gpu(const float* X_host, float* W_host, float* P_host, const int n,
              const int m, const int k, const int n_iter, int batch_size,
              const float etaW, const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int noise_type, const int activation_type, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed, int gpu_id);

int train_gpu_sparse(const sparseMatrix* X_host, float* W_host, float* P_host, const int n,
              const int m, const int k, const int n_iter, int batch_size,
              const float etaW, const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int noise_type, const int activation_type, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed, int gpu_id);

/**
 * Given a trained RFN, this will calculate the weights that are used to
 * estimate the hidden activations.
 *
 * This needs access to the training data, as the W need to incorporate
 * the scaling that would otherwise be done on the hidden activations.
 * The scaling parameters have to be fitted on the training data's H.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory. All transfers from and to the GPU will be
 * done internally by the function itself.
 *
 * @param X             [n, m] training data matrix, with 1 sample per row
 * @param W             [k, m] RFN weight matrix
 * @param P             [m] vector, contains Psi
 * @param Wout          [k, m] output weight matrix
 * @param n             number of training samples
 * @param m             number of input features
 * @param k             number of hidden units
 * @param gpu_id        ID of the GPU that this will run on (if this is -1,
 *                      the GPU with the most free memory will be picked)
 */
void calculate_W_gpu(const float* X, const float* W, const float* P, float* Wout,
                     const int n, const int m, const int k,
                     const int activation_type, const int apply_scaling, const float h_threshold,
                     int gpu_id);
void calculate_W_gpu_sparse(const sparseMatrix* X, const float* W, const float* P, float* Wout,
                     const int n, const int m, const int k,
                     const int activation_type, const int apply_scaling, const float h_threshold,
                     int gpu_id);
#endif

#ifdef __cplusplus
}
#endif

#endif /* LIBRFN_H */
