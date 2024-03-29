#include "clutil.h"
#include "fft.h"
#include "kernels.h"

static unsigned workOffset[MAX_GPU_COUNT];
static unsigned workSize[MAX_GPU_COUNT];

bool
runSande_tookeyFFT(const char* const argv[], const unsigned n,
                                         const unsigned size)
{
  if (!initExecution(size, n)) {
         return false;
    }
    sande_tookeyFFTGpu(argv, n, size);
    return true;
}




int
sande_tookeyFFTGpu(const char* const argv[], const unsigned n,
                                             const unsigned size)

{
    const unsigned powN = (unsigned)log2(n);
    printf("Compiling sande-tookey Program..\n");
    compileProgram(argv, "fft.h", "kernels/sande_tookey.cl");

    printf("Creating Kernel\n");
    for (unsigned i = 0; i < deviceCount; ++i) {
        createKernel(i, "sande_tookey");
    }

    const unsigned sizePerGPU = size / deviceCount;
    for (unsigned i = 0; i < deviceCount; ++i) {
        workSize[i] = (i != (deviceCount - 1)) ? sizePerGPU 
                                               : (size - workOffset[i]);       
        
        allocateDeviceMemory(i , workSize[i], workOffset[i]);
        
        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void*) &d_Freal[i]);
        clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void*) &d_Fimag[i]);
	clSetKernelArg(kernel[i], 2, sizeof(unsigned), &n);
	clSetKernelArg(kernel[i], 3, sizeof(unsigned), &powN);
	clSetKernelArg(kernel[i], 4, sizeof(unsigned), &blockSize);
       

        if ((i + 1) < deviceCount) {
            workOffset[i + 1] = workOffset[i] + workSize[i];
        } 

    }
    
    size_t localWorkSize[] = {blockSize};
    //size_t globalWorkSize[]= {shrRoundUp(BLOCK_SIZE, ARR_SIZE + BLOCK_SIZE -1)};

    for (unsigned i = 0; i < deviceCount; ++i) {
        size_t globalWorkSize[] = {shrRoundUp(blockSize, workSize[i])}; 
        // kernel non blocking execution 
        runKernel(i, localWorkSize, globalWorkSize);
    }

    for (unsigned i = 0; i < deviceCount; ++i) {
        copyFromDevice(i, d_Freal[i], h_Rreal + workOffset[i],
                                                workSize[i]); 
        copyFromDevice(i, d_Fimag[i], h_Rimag + workOffset[i],
                                                 workSize[i]);
    }

    // wait for copy event
    const cl_int ciErrNum = clWaitForEvents(deviceCount, gpuDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");

    printGpuTime();
    return 1;
}
