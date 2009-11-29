#include <omp.h>
#include "clutil.h"
#include "fft.h"
#include "kernels.h"

static unsigned workOffset[MAX_GPU_COUNT];
static unsigned workSize[MAX_GPU_COUNT];

int
runSlowFFT(const char* const argv[], const unsigned n, 
                                  const unsigned size)
{
     //TODO:: split the work   
     #pragma omp parallel for
     for (unsigned i = 0; i < 2; ++i) {
         if ( i == 0) {
             slowFFTGpu(argv, n,  size / 2);
         } else {
             printf("Running on CPU..\n");
             const unsigned start = size / 2;
             slowFFTCpu(start, n, size / 2);
         }      
     }
     printf("Results : \n");
     for (unsigned i = 0; i < size; ++i) {
        printf("%f + i%f \n", h_Rreal[i], h_Rimag[i]);
     }
     return 1;
}

// FOR GPU
int
slowFFTGpu(const char* const argv[], const unsigned n,
                                             const unsigned size)

{
    printf("Compiling slowFFT Program for GPU..\n");
    compileProgram(argv, "fft.h", "kernels/slowfft.cl");

    for (unsigned i = 0; i < deviceCount; ++i) {
        createKernel(i, "slowfft");
    }

    const unsigned sizePerGPU = size / deviceCount;
    for (unsigned i = 0; i < deviceCount; ++i) {
        workSize[i] = (i != (deviceCount - 1)) ? sizePerGPU 
                                               : (size - workOffset[i]);       
        
        allocateDeviceMemory(i , workSize[i], workOffset[i]);
        
        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void*) &d_Freal[i]);
        clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void*) &d_Fimag[i]);
        clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void*) &d_Rreal[i]);
        clSetKernelArg(kernel[i], 3, sizeof(cl_mem), (void*) &d_Rimag[i]);
        clSetKernelArg(kernel[i], 4, sizeof(unsigned), &n);
        clSetKernelArg(kernel[i], 5, sizeof(unsigned), &blockSize);
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
        copyFromDevice(i, d_Rreal[i], h_Rreal + workOffset[i],
                                                workSize[i]); 
        copyFromDevice(i, d_Rimag[i], h_Rimag + workOffset[i],
                                                 workSize[i]);
    }

    // wait for copy event
    const cl_int ciErrNum = clWaitForEvents(deviceCount, gpuDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");

    return 1;
}

// FOR CPU

void
slowFFTCpu(const unsigned offset, const unsigned N, const unsigned  size)
{
    const float ph = ( -1 *2.0 * 3.14159265359) / N;
    #pragma omp parallel for
    for (unsigned i = 0; i <  size / N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            const unsigned index = i * N + j + offset ;
            const unsigned start = i * N + offset;
            const unsigned end = (i + 1) * N + offset;
            float real = 0;
            float imag = 0;
            for (unsigned k = start; k < end; ++k) {
                const float rx = h_Freal[k];
                const float ix = h_Fimag[k];
                const float val = ph * (k % N) *j;
                real +=  rx* cos(val) - ix * sin(val);
                imag +=  rx* sin(val) + ix * cos(val);
            }

            h_Rreal[index] = real;
            h_Rimag[index] = imag;
        }
    }
}
