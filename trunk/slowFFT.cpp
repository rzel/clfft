#include "clutil.h"
#include "fft.h"


int
slowFFT( const char* const argv[], const unsigned n, const int is,
                                               const unsigned size)

{

    allocateDeviceMemory(size);
    printf("Compiling Program..\n");
    compileProgram(argv, "fft.h", "kernels/slowfft.cl", 1);
    createKernel("slowfft");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &d_Freal);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &d_Fimag);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &d_Rreal);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &d_Rimag);
    clSetKernelArg(kernel, 4, sizeof(unsigned), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &is);
    

    size_t localWorkSize[] = {BLOCK_SIZE};
    size_t globalWorkSize[]= {shrRoundUp(BLOCK_SIZE, ARR_SIZE + BLOCK_SIZE -1)};
    runKernel(kernel, 1, /* Work Dimension. */
              localWorkSize,
              globalWorkSize);

    copyFromDevice(d_Rreal, sizeof(float) * ARR_SIZE, h_Rreal, true);
    copyFromDevice(d_Rimag, sizeof(float) * ARR_SIZE, h_Rimag, true);

    printf("Results : \n");
    for (unsigned i = 0; i < ARR_SIZE; ++i) {
        printf("%f + i%f \n", h_Rreal[i], h_Rimag[i]);
    }
    return 1;
}


