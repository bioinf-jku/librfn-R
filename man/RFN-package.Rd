\name{RFN-package}
\alias{RFN-package}
\alias{rfn}
\alias{RFN}
\docType{package}
\title{
Rectified Factor Networks
}
\description{
This package implements the Rectified Factor Networks algorithm 
as described by Clevert et al., 2014, which is an unsupervised 
latent variable model.
}
\details{
\tabular{ll}{
Package: \tab RFN\cr
Type: \tab Package\cr
Version: \tab 0.1\cr
Date: \tab 2016-02-19\cr
License: \tab GPL (>= 2)\cr
}
The core of this package is the function \code{train_rfn}, which trains a Rectified Factor Network 
as described by Clevert et al., 2014. Alongside, there exist three basic functions for plotting data, 
particularly MNIST pictures, implemented in \code{showSinglePicture}, \code{showSingleColorPicture} 
and \code{showPictures}. 
}
\author{
Thomas Unterthiner, Thomas Adler

Maintainer: Thomas Unterthiner <unterthiner@bioinf.jku.at>
}
\references{
Clevert DA, Unterthiner T, Mayr A, Ramsauer H, Hochreiter S, "Rectified Factor Networks", http://arxiv.org/abs/1502.06464
}

