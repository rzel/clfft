#include "clutil.h"
#include "fft.h"

int
cooleyTukey( const char* const argv[], float* hFreal, float* hFimag, 
            float* hRreal, float* hRimag,
            const unsigned n) 
{
    const cl_mem d_Freal = createDeviceBuffer(
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * ARR_SIZE,
                        hFreal,
                        true);

    const cl_mem d_Fimag = createDeviceBuffer(
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * ARR_SIZE,
                        hFimag,
                        true);

    const cl_mem d_Rreal = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * ARR_SIZE,
                                              hRreal,
                                              true);

    const cl_mem d_Rimag = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * ARR_SIZE,
                                              hRimag,
                                              true);
    const unsigned powN = (unsigned)log2(n);

    printf("Compiling Program..\n");
    const cl_program cpProgram = compileProgram(argv, "fft.h", 
                                                "kernels/cooleyTukey.cl", 1);

    const cl_kernel kernobj = createKernel(cpProgram,"reverse");

    clSetKernelArg(kernobj, 0, sizeof(cl_mem), (void*) &d_Freal);
    clSetKernelArg(kernobj, 1, sizeof(cl_mem), (void*) &d_Fimag);
    clSetKernelArg(kernobj, 2, sizeof(cl_mem), (void*) &d_Rreal);
    clSetKernelArg(kernobj, 3, sizeof(cl_mem), (void*) &d_Rimag);
    clSetKernelArg(kernobj, 4, sizeof(unsigned), &n); 
    clSetKernelArg(kernobj, 5, sizeof(unsigned), &powN);

    size_t localWorkSize[] = {BLOCK_SIZE};
    size_t globalWorkSize[]= {shrRoundUp(BLOCK_SIZE, ARR_SIZE + BLOCK_SIZE -1)};
    
    runKernel(kernobj, 1, /* Work Dimension. */
              localWorkSize,
              globalWorkSize);

    copyFromDevice(d_Freal, sizeof(float) * ARR_SIZE, hRreal, true);
    copyFromDevice(d_Fimag, sizeof(float) * ARR_SIZE, hRimag, true);

    for (unsigned i = 0; i < ARR_SIZE; ++i) {
        printf("%f + i%f \n", hRreal[i], hRimag[i]);
    }
    return 1;
}

