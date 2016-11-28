library(Rcpp)

#' Trains a Rectified Factor Network (RFN)
#' 
#' Trains an RFN as described by Clevert et al., 2014
#' @param X The data matrix
#' @param n_hidden Number of latent variables to estimate
#' @param n_iter Number of iterations to run the algorithm
#' @param etaW Learning rate of the W parameter
#' @param etaP Learning rate of the Psi parameter (It's probably save to set this to the same value as etaW)
#' @param minP Minimal value for Psi. Should be in 1e-5 - 1e-1
#' @param batch_size If > 2, this will activate mini-batch learning instead of full batch learning
#' @param dropout_rate Dropout rate for the latent variables
#' @param input_noise_rate Noise/dropout rate for input variables
#' @param l2_weightdecay L2 penalty for weight decay
#' @param l1_weightdecay L1 penalty for weight decay
#' @param h_threshold Threshhold for rectifying/leaky activations
#' @param momentum Momentum term for learning
#' @param noise_type Type of input noise. One of "dropout", "saltpepper", "gaussian"
#' @param activation Activation function for hidden/latent variables. One of "linear", "relu", "leaky", "sigmoid", "tanh"
#' @param apply_scaling Scale the data
#' @param apply_newton_update Whether to use a Newton update (default) or a gradient descent step
#' @param seed Seed for the random number generator
#' @param use_gpu Use the gpu (default cpu). Works only for sparse input.
#' @param gpu_id If use_gpu is true, use gpu with this id (default -1 selects one available)
#' @return Returns a list of matrices \code{W}, \code{P}, \code{H}, 
#'  \code{Wout}, whereas \code{W \%*\% H} is the noise-free reconstruction 
#'  of the data \code{X} and \code{diag(P)} is the covariance matrix 
#'  of the additive zero-mean noise. \code{Wout} is the matrix that 
#'  maps input vectors to their latent representation, usually by 
#'  \code{pmax(t(Wout) \%*\% X, 0)}.
#' @export
#' @useDynLib RFN
#' @importFrom Rcpp evalCpp
train_rfn <- function(X, n_hidden, n_iter, etaW, etaP, minP, batch_size=-1,
   dropout_rate=0.0, input_noise_rate=0.0, l2_weightdecay=0.0,
   l1_weightdecay=0.0, h_threshold=0.0, momentum=0.0, noise_type="saltpepper",
   activation="relu", apply_scaling=1, apply_newton_update=1, seed=-1, use_gpu=0, gpu_id=-1)
{
   if (is.data.frame(X))
      X <- data.matrix(X)
    
   if (!(is(X, "dgCMatrix") || is.numeric(X)) || length(dim(X)) != 2 || sum(is.na(X)) != 0)
      stop("X must be a numeric matrix without NAs")
   
   if (!is.numeric(c(n_hidden, n_iter, etaW, etaP, minP, batch_size, dropout_rate, input_noise_rate, 
      l2_weightdecay, l1_weightdecay, h_threshold, momentum, apply_scaling, apply_newton_update, 
      seed)))
      stop("at least one of the numeric params is not numeric")
   
   noise_type <- switch(noise_type, dropout=1, saltpepper=2,gaussian=3)
   
   if (is.null(noise_type))
      stop("noise_type must be one of \"dropout\", \"saltpepper\", \"gaussian\"")
   
   activation <- switch(activation, linear=0, relu=1, leaky=2, sigmoid=3, tanh=4)
   
   if (is.null(activation))
      stop("activation must be one of \"linear\", \"relu\", \"leaky\", \"sigmoid\", \"tanh\"")
   
   set.seed(ifelse(seed < 0, sample.int(10000, size=1), seed));
   
   n <- ncol(X)
   m <- nrow(X)
   W <- matrix(0.01*rnorm(n_hidden * m), ncol=n_hidden)
   P <- rep(0.1, m)
   
   if (is(X, "dgCMatrix"))
   {
      require(Matrix)
      #tX <- Matrix::t(X) # convert X from colmajor to rowmajor
      
      if (noise_type == 3)
         stop("cannot use Gaussian noise on sparse input matrix")
     
     if (use_gpu)
     {
       res1 <- .Call('train_rfn_gpu_sparse', X@x, X@p, X@i, W, P, as.integer(n), as.integer(m), as.integer(n_hidden), 
                     as.integer(n_iter), as.integer(batch_size), etaW, etaP, minP, h_threshold, dropout_rate, 
                     input_noise_rate, l2_weightdecay, l1_weightdecay, momentum, as.integer(noise_type), 
                     as.integer(activation), as.integer(apply_scaling), as.integer(apply_newton_update),
                     as.integer(seed), as.integer(gpu_id), PACKAGE = 'RFN')
       
       Wout <- .Call('calculate_rfn_W_gpu_sparse', X@x, X@p, X@i, res1$W, res1$P, as.integer(n), as.integer(m), 
                     as.integer(n_hidden), as.integer(activation), as.integer(apply_scaling), h_threshold,
                     as.integer(gpu_id), PACKAGE = 'RFN')
     }
     else
     {
        res1 <- .Call('train_rfn_cpu_sparse', X@x, X@p, X@i, W, P, as.integer(n), as.integer(m), as.integer(n_hidden), 
           as.integer(n_iter), as.integer(batch_size), etaW, etaP, minP, h_threshold, dropout_rate, 
           input_noise_rate, l2_weightdecay, l1_weightdecay, momentum, as.integer(noise_type), 
           as.integer(activation), as.integer(apply_scaling), as.integer(apply_newton_update),
           as.integer(seed), PACKAGE = 'RFN')
      
        Wout <- .Call('calculate_rfn_W_sparse', X@x, X@p, X@i, res1$W, res1$P, as.integer(n), as.integer(m), 
          as.integer(n_hidden), as.integer(activation), as.integer(apply_scaling), h_threshold,
          PACKAGE = 'RFN')
     }
   }
   else
   {
      #tX <- t(X) # convert X from colmajor to rowmajor
      if (use_gpu)
      {
        res1 <- .Call('train_rfn_gpu', X@x, X@p, X@i, W, P, as.integer(n), as.integer(m), as.integer(n_hidden), 
                     as.integer(n_iter), as.integer(batch_size), etaW, etaP, minP, h_threshold, dropout_rate, 
                     input_noise_rate, l2_weightdecay, l1_weightdecay, momentum, as.integer(noise_type), 
                     as.integer(activation), as.integer(apply_scaling), as.integer(apply_newton_update),
                     as.integer(seed), as.integer(gpu_id), PACKAGE = 'RFN')
       
        Wout <- .Call('calculate_rfn_W_gpu', X@x, X@p, X@i, res1$W, res1$P, as.integer(n), as.integer(m), 
                     as.integer(n_hidden), as.integer(activation), as.integer(apply_scaling), h_threshold,
                     as.integer(gpu_id), PACKAGE = 'RFN')
      }
      else 
      {
        res1 <- .Call('train_rfn_cpu', X, W, P, as.integer(n), as.integer(m), as.integer(n_hidden), 
           as.integer(n_iter), as.integer(batch_size), etaW, etaP, minP, h_threshold, dropout_rate, 
           input_noise_rate, l2_weightdecay, l1_weightdecay, momentum, as.integer(noise_type), 
           as.integer(activation), as.integer(apply_scaling), as.integer(apply_newton_update),
           as.integer(seed), PACKAGE = 'RFN')
      
        Wout <- .Call('calculate_rfn_W', X, res1$W, res1$P, as.integer(n), as.integer(m), 
           as.integer(n_hidden), as.integer(activation), as.integer(apply_scaling), h_threshold,
           PACKAGE = 'RFN')
      }
   }
   
   # convert results from rowmajor to colmajor
   #res1$W <- t(res1$W)
   #dim(res1$W) <- c(n_hidden, m)
   #Wout <- t(Wout)
   #dim(Wout) <- c(n_hidden, m)
   
   H <- t(Wout) %*% X
   H <- pmax(as.matrix(H), h_threshold)
   return(list(W=res1$W, P=res1$P, H=H, Wout=Wout, T=res1$T))
}


#' Displays a picture, such as an MNIST digit or CIFAR image
#' Assumes the image is stored in a rowwise-fashion in the vector x
#' @param x the data to depict, a grayscale channel in \code{showSinglePicture}, or in 
#' \code{showSingleColorPicture} it is assumed to be a vector where the first
#' third of the elements describe the red color channel, the next ones
#' the green channel and the last third the blue channel; in \code{showPictures} its interpretation 
#' is controlled by the parameter \code{isColor}.
#' @param sidelength the height of the resulting picure in pixels; on \code{NULL}, a square picture
#'  is assumed
#' @param col as passed to function \code{image}, see \code{?image}
#' @param ... as passed to function \code{image}, see \code{?image}
#' @return \code{NULL}
#' @rdname showPictures
#' @export
showSinglePicture <- function(x, sidelength=NULL, col=gray(256:1/256), ...) {
	if (is.null(sidelength))
		sidelength <- floor(sqrt(length(x)) + 0.5)

	p <- par(mar = c(0, 0, 0, 0))

	# useRaster=TRUE looks much better in Latex
	image(t(matrix(x, nrow = sidelength, byrow = TRUE)), col = col,
			ylim = c(1, 0), yaxt='n', xaxt='n', ann=FALSE, useRaster=TRUE, ...)
	par(p)
	invisible(NULL)
}


#' Displays a color picture.
#'
#' Assumes the image is stored in a rowwise-fashion.
#' @param normalize logical, whether \code{x} should be normalized
#' @rdname showPictures
#' @export
showSingleColorPicture <- function(x, sidelength=NULL, normalize=TRUE, ...) {
	if (is.null(sidelength))
		sidelength <- as.integer(floor(sqrt(length(x)/3) + 0.5))

	if (normalize) {
		norm01 <- function(x)  (x - min(x)) / (max(x) - min(x)+1e-10)
		x <- norm01(x)

		# conversion function from pylearn2
		#norm01 <- function(y)  (0.5*(y / max(abs(y)))+0.5)
		#r = norm01(x[1:1024])
		#b = norm01(x[1025:2048])
		#g = norm01(x[2049:3072])
	}

	ss <- sidelength*sidelength
	r = x[1:ss]
	g = x[(ss+1):(2*ss)]
	b = x[(2*ss+1):(3*ss)]
	colvec <- rgb(r, g, b)
	colors <- unique(colvec)
	colmat <- array(match(colvec, colors), dim=c(sidelength, sidelength))

	p <- par(mar = c(0, 0, 0, 0))
	image(x = 0:sidelength, y=0:sidelength, z = colmat, col=colors,
			ylim = c(sidelength, 0), yaxt='n', xaxt='n', ann=FALSE, useRaster=TRUE)
	par(p)
	invisible(NULL)
}

#' Shows multiple colored or grayscale pictures
#' @param data contains the pixel data, one picture in each column
#' @param nrow,ncol define the arrangement of the pictures
#' @param idx defines the enumeration of the pictures, i.e. the order in which the columns of 
#'  \code{data} should be processed; on \code{NULL}, 1,2,3... is assumed
#' @param isColor logical, indicates whether \code{data} is to be interpreted as single channel
#'  or 3-way-channel (RGB)
#' @rdname showPictures
#' @export
showPictures <- function(data, nrow, ncol, idx=NULL, isColor=FALSE, ...) {
	if (is.null(idx)) {
		m <- min(ncol(data), nrow*ncol)
		idx <- 1:m
	}

	# small hack: if we have a sidelength of 28, we probably display
	# MNIST digits, which are easier to read with inverted colors
	col = gray(0:255/255)
	if (floor(sqrt(length(data[, 1])) + 0.5) == 28)
		col = gray(255:0/255)

	par(mfrow=c(nrow, ncol))
	for (i in idx) {
		if (isColor)
			showSingleColorPicture(data[, i], ...)
		else
			showSinglePicture(data[, i], col=col, ...)
	}
	par(mfrow=c(1, 1))
	invisible(NULL)
}

#' for testing MKL
#' @param A a matrix
#' @rdname somatcopy
#' @export
somatcopy <- function(A)
{
   .Call('somatcopy', nrow(A), ncol(A), A)
}

#' for testing MKL
#' @param A a sparse matrix
#' @param B a matrix
#' @rdname somatcopy
#' @export
gemm <- function(A, B)
{
   require(Matrix)
   tA <- Matrix::t(A)
   C <- .Call('gemm', 'r', 'n', 't', nrow(A), ncol(A), ncol(B), tA@x, tA@i, tA@p, t(B))
   attributes(C)$dim <- rev(attributes(C)$dim)
   C <- t(C)
}



