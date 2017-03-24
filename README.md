# librfn-R: R extension for Rectified Factor Networks

Rectified Factor Networks (RFNs) are an unsupervised technique that learns a non-linear, high-dimensional representation of its input. The underlying algorithm has been published in

*Rectified Factor Networks*, Djork-Arné Clevert, Andreas Mayr, Thomas Unterthiner, Sepp Hochreiter, NIPS 2015.

librfn is implemented in C++ and can be easily integrated in existing code bases. It also contains a high-level Python wrapper for ease of use. The library can run in either CPU or GPU mode. For larger models the GPU mode offers large speedups and is the recommended mode.

librfn has been written by [Thomas Unterthiner](http://www.bioinf.jku.at/people/unterthiner/) and [Djork-Arné Clevert](http://www.bioinf.jku.at/people/clevert/). Sparse matrix support was added by Balázs Bencze and Thomas Adler.

# Installation

Type the following in the parent directory of the repository. The parameter --configure-args is optional and can be omitted.
R CMD INSTALL \
  --configure-args=' \
  --with-cuda-home=/usr/local/cuda \
  --with-mkl-home=/opt/intel/mkl' \
  librfn-R


# Requirements
To run the GPU code, you require a CUDA 7.5 (or higher) compatible GPU. While in theory CUDA 7.0 is also supported, it contains a bug that results in a memory leak when running librfn (and your program is likely to crash with an out-of-memory error).


# Implementation Note

The RFN algorithm is based on the EM algorithm. Within the E-step, the published algorithm includes a projection procedure that can be implemented in several ways (see the RFN paper's supplemental section 9). To make sure no optimzation constraints are violated during this projection, the original publication tries the simplest method first, but backs out to more and more complicated updates if easier method fail (suppl. section 9.5.3).
In contrast, librfn always uses the simplest/fastest projection method. This is a simplification/approximation of the original algorithm that nevertheless works very well in practice.


# License
librfn was developed by Thomas Unterthiner and is licensed under the [General Public License (GPL) Version 2 or higher](http://www.gnu.org/licenses/gpl-2.0.html) See ``License.txt`` for details.
