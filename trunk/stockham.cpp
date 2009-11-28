#include "clutil.h"
#include "fft.h"

static unsigned workOffset[MAX_GPU_COUNT];
static unsigned workSize[MAX_GPU_COUNT];


int
stockhamFFT(const char* const argv[], const unsigned n, const int is,
                                             const unsigned size)

{
    const unsigned powN = (unsigned)log2(n);
    printf("Compiling Program..\n");
    compileProgram(argv, "fft.h", "kernels/stockham.cl");

    printf("Creating Kernel\n");
    for (unsigned i = 0; i < deviceCount; ++i) {
        createKernel(i, "stockham");
    }


    const unsigned sizePerGPU = size / deviceCount;
    for (unsigned i = 0; i < deviceCount; ++i) {
        workSize[i] = (i != (deviceCount - 1)) ? sizePerGPU 
                                               : (size - workOffset[i]);       
        
        printf("Allocating device memory for device %d\n", i);
        allocateDeviceMemory(i , workSize[i], workOffset[i]);
        
        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void*) &d_Rreal[i]);
        clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void*) &d_Rimag[i]);
	clSetKernelArg(kernel[i], 2, sizeof(int), &is);
	clSetKernelArg(kernel[i], 3, sizeof(unsigned), &n);
	clSetKernelArg(kernel[i], 5, sizeof(unsigned), &powN);

        if ((i + 1) < deviceCount) {
            workOffset[i + 1] = workOffset[i] + workSize[i];
        } 

    }
    
    size_t localWorkSize[] = {BLOCK_SIZE};
    //size_t globalWorkSize[]= {shrRoundUp(BLOCK_SIZE, ARR_SIZE + BLOCK_SIZE -1)};

    for (unsigned i = 0; i < deviceCount; ++i) {
        size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, workSize[i])}; 
        // kernel non blocking execution 
        runKernel(i, localWorkSize, globalWorkSize);
    }

    for (unsigned i = 0; i < deviceCount; ++i) {
        copyFromDevice(i, d_Rreal[i], h_Rreal + workOffset[i],
                                                workSize[i]); 
        copyFromDevice(i, d_Rimag[i], h_Rimag + workOffset[i],
                                                 workSize[i]);
    }

    // wait for copy event
    const cl_int ciErrNum = clWaitForEvents(deviceCount, gpuDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");

    printf("Results : \n");
    for (unsigned i = 0; i < ARR_SIZE; ++i) {
        printf("%f + i%f \n", h_Rreal[i], h_Rimag[i]);
    }
    return 1;
}
