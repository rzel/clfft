#include "clutil.h"
#include "fft.h"


int
slowFFT( const char* const argv[], float* h_Freal, float* h_Fimag,
         float* h_Rreal, float* h_Rimag,
         const unsigned n, const int is)

{
    const cl_mem d_Freal = createDeviceBuffer(
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * ARR_SIZE,
                        h_Freal,
                        true);

    const cl_mem d_Fimag = createDeviceBuffer(
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * ARR_SIZE,
                        h_Fimag,
                        true);

    const cl_mem d_Rreal = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * ARR_SIZE,
                                              h_Rreal,
                                              true);

    const cl_mem d_Rimag = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * ARR_SIZE,
                                              h_Rimag,
                                              true);


    printf("Compiling Program..\n");
    const cl_program cpProgram = compileProgram(argv, "fft.h", "kernels/slowfft.cl", 1);

    const cl_kernel kernobj = createKernel(cpProgram,"slowfft");

    clSetKernelArg(kernobj, 0, sizeof(cl_mem), (void*) &d_Freal);
    clSetKernelArg(kernobj, 1, sizeof(cl_mem), (void*) &d_Fimag);
    clSetKernelArg(kernobj, 2, sizeof(cl_mem), (void*) &d_Rreal);
    clSetKernelArg(kernobj, 3, sizeof(cl_mem), (void*) &d_Rimag);
    clSetKernelArg(kernobj, 4, sizeof(unsigned), &n);
    clSetKernelArg(kernobj, 5, sizeof(int), &is);
    

    size_t localWorkSize[] = {BLOCK_SIZE};
    size_t globalWorkSize[]= {shrRoundUp(BLOCK_SIZE, ARR_SIZE + BLOCK_SIZE -1)};
    runKernel(kernobj, 1, /* Work Dimension. */

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


