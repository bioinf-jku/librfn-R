## This scipt is a modification of the configure.ac file
## from the permGPU package (version 0.14.9)
## http://cran.r-project.org/web/packages/permGPU/
## Licensed under GNU GPL 3 or later

AC_INIT(sparse-rfn, version-0.1)
## Cuda Toolkit

AC_SUBST([CUDA_HOME])
AM_CONDITIONAL([HAS_CUDA_HOME], [test ! -z "$CUDA_HOME"])
AM_COND_IF([HAS_CUDA_HOME],[],[AC_ARG_WITH([cuda-home],
	[AC_HELP_STRING([--with-cuda-home=DIR],
		[full path to CUDA installation @<:@/usr/local/cuda@:>@])],
	[CUDA_HOME="$withval"], [CUDA_HOME=/usr/local/cuda])])

NVCC=${CUDA_HOME}/bin/nvcc
AC_CHECK_FILE([${NVCC}],,[AC_MSG_ERROR([nvcc not found!])])

CUDA_INCLUDE=${CUDA_HOME}/include
AC_CHECK_FILE([${CUDA_INCLUDE}/cublas.h],,[AC_MSG_ERROR([CUDA headers not found!])])

AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [check_bit=64-bit])
if test "${check_bit}" == "64-bit"; then
	CUDA_LD_LOC="-L${CUDA_HOME}/lib64"
else
	CUDA_LD_LOC="-L${CUDA_HOME}/lib"
fi
CUDA_LD="${CUDA_LD_LOC} -lcublas -lcurand -lcuda -lcudart -lcusolver -lcusparse"

## MKL
AC_SUBST([MKL_HOME])
AM_CONDITIONAL([HAS_MKL_HOME], [test ! -z "$MKL_HOME"])
AM_COND_IF([HAS_MKL_HOME],[],[AC_ARG_WITH([mkl-home],
	[AC_HELP_STRING([--with-mkl-home=DIR],
		[full path to MKL installation @<:@/opt/intel/mkl@:>@])],
	[MKL_HOME="$withval"], [MKL_HOME=/opt/intel/mkl])])

MKL_INCLUDE=${MKL_HOME}/include
AC_CHECK_FILE([${MKL_INCLUDE}/mkl_blas.h],[has_mkl_blas_header=yes])
if test "${has_mkl_blas_header}" == "yes"; then
	AC_CHECK_FILE([${MKL_HOME}/lib],[has_mkl_lib=yes])
	if test "${has_mkl_lib}" == "yes"; then
		AC_CHECK_FILE([${MKL_HOME}/lib/intel64/libmkl_rt.so], [check_mkl_bit=64-bit])
		if test "${check_mkl_bit}" == "64-bit"; then
			MKL_LIB_LOC="${MKL_HOME}/lib/intel64"
		else
			MKL_LIB_LOC="${MKL_HOME}/lib/ia32"
		fi
		CPPFLAGS="-I${MKL_INCLUDE} -DMKL"
		BL_LIBS="-L${MKL_LIB_LOC} -lmkl_rt"
	else
		CPPFLAGS=""
		BL_LIBS="\$(BLAS_LIBS) \$(LAPACK_LIBS) \$(FLIBS)"
	fi
else
	CPPFLAGS=""
	BL_LIBS="\$(BLAS_LIBS) \$(LAPACK_LIBS) \$(FLIBS)"
fi

AC_SUBST([NVCC])
AC_SUBST([BL_LIBS])
AC_SUBST([CUDA_LD])
AC_SUBST([CUDA_INCLUDE])
AC_SUBST([CPPFLAGS])

AC_CONFIG_FILES([src/Makevars:Makevars.in])
AC_OUTPUT

# Delete Makefile from librfn
rm -f src/Makefile
