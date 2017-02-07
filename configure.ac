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
AC_MSG_CHECKING([for CUDA headers])
AC_CHECK_FILE([${CUDA_INCLUDE}/cublas.h],,
	[AC_MSG_ERROR([CUDA headers not found!])]
)

AC_MSG_CHECKING([whether this is a 64-bit version of CUDA])
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [check_bit=64-bit])
if test "${check_bit}" == "64-bit"; then
	CUDA_LD="-L${CUDA_HOME}/lib64"
else
	CUDA_LD="-L${CUDA_HOME}/lib"
fi

AC_SUBST([NVCC])
AC_SUBST([CUDA_LD])
AC_SUBST([CUDA_INCLUDE])

AC_CONFIG_FILES([src/Example])
AC_OUTPUT
