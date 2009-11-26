#include "clutil.h"
#include "fft.h"

static unsigned workOffset[MAX_GPU_COUNT];
static unsigned workSize[MAX_GPU_COUNT];

int
cooleyTukey(const char* const argv[], const unsigned n, const unsigned size) 
{
    const unsigned powN = (unsigned)log2(n);

    printf("Compiling Cooley Tukey Program..\n");
    compileProgram(argv, "fft.h", "kernels/cooleyTukey.cl");

    for (unsigned i = 0; i < deviceCount; ++i) {
        createKernel(i, "reverse");
    }
    
    const unsigned sizePerGPU = size / deviceCount;
    int Iter =0;
    float lIter =0;
	cl_int cerrNum =CL_SUCCESS;
        const cl_mem d_ter = clCreateBuffer(cxContext,CL_MEM_WRITE_ONLY ,sizeof(float),&lIter,&cerrNum);

    for (unsigned i = 0; i < deviceCount; ++i) {
        workSize[i] = (i != (deviceCount - 1)) ? sizePerGPU
                                               : (size - workOffset[i]);

        printf("Allocating device memory for device %d\n", i);
        allocateDeviceMemory(i , workSize[i], workOffset[i]);
	printf("Before \n");
	checkError(cerrNum,CL_SUCCESS,"clCreateBuffer");
        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void*) &d_Freal[i]);
        clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void*) &d_Fimag[i]);
        clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void*) &d_Rreal[i]);
        clSetKernelArg(kernel[i], 3, sizeof(cl_mem), (void*) &d_Rimag[i]);
        clSetKernelArg(kernel[i], 4, sizeof(unsigned), &n); 
        clSetKernelArg(kernel[i], 5, sizeof(unsigned), &powN);
        clSetKernelArg(kernel[i], 6, sizeof(cl_mem), (void*) &d_ter);

        if ((i + 1) < deviceCount) {
            workOffset[i + 1] = workOffset[i] + workSize[i];
        }

    }

    size_t localWorkSize[] = {BLOCK_SIZE};
   
    for (unsigned i = 0; i < deviceCount; ++i) { 
        size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, workSize[i])};
        runKernel(i, localWorkSize, globalWorkSize);
    }

    for (unsigned i = 0; i < deviceCount; ++i) {   
         copyFromDevice(i, d_Rreal[i], h_Rreal + workOffset[i],
                                                 workSize[i]);
         copyFromDevice(i, d_Rimag[i], h_Rimag + workOffset[i],
			 workSize[i]);
	 copyFromDevice(i, d_Fimag[i], h_Fimag + workOffset[i],
			 workSize[i]);
	 copyFromDevice(i, d_Freal[i], h_Freal + workOffset[i],
			 workSize[i]);

    }

    cl_int cErr = clEnqueueReadBuffer(commandQueue[0],d_ter,CL_FALSE,0,sizeof(float),&lIter,0,NULL,&gpuDone[0]);

   
    // wait for copy event
    const cl_int ciErrNum = clWaitForEvents(deviceCount, gpuDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");

    for (unsigned i = 0; i < ARR_SIZE; ++i) {
        printf("%f + i%f \n", h_Rreal[i], h_Rimag[i]);
    }

    printf("The Second array\n");
    for (unsigned i = 0; i < ARR_SIZE; ++i) {
	    printf("%f + i%f \n", h_Freal[i], h_Fimag[i]);
    }
    printf("The Number of Iterations = %f powN  = %d\n",lIter,powN);

    return 1;
}

