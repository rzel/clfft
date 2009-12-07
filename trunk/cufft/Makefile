CUDA_INSTALL_PATH ?= /opt/cuda
CUDA_SDK_PATH ?= /opt/cuda-sdk/

NVCC    := $(CUDA_INSTALL_PATH)/bin/nvcc
NVCCFLAGS := --ptxas-options=-v 

LIBPATH   := -L$(CUDA_SDK_PATH)/lib64

LIBS    := -lcutil -lcufft
INCLUDES  +=  -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc

all:
	$(NVCC) simpleCUFFT.cu  $(NVCCFLAGS) $(LIBPATH) $(LIBS) $(INCLUDES)


clean:
	rm a.out
