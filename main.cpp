
#include "clutil.h"

#include "vecAdd.h"

int 
main(const int argc, const char* argv[])
{
    size_t localWorkSize[] = {BLOCK_SIZE};
    size_t globalWorkSize[]= {shrRoundUp(BLOCK_SIZE, ARR_SIZE + BLOCK_SIZE -1)};
    
    if(argc < 2)
      {
	printf("Usage: clfft N [i]\n");
	return 0;
      }

    int n = atoi(argv[1]);
    int is = 1;
    if(argc == 3)		// This is a weak check, but should do for now.
      {
	is = -1;
      }

    printf("Initializing CL Context..\n");
    if (init_cl_context(CL_DEVICE_TYPE_GPU) !=CL_SUCCESS) {
        printf("Error ! Aborting..\n");
        exit(1);
    }

    printf("Getting Device Count..\n");
    cl_uint ciDeviceCount;
    getDeviceCount(ciDeviceCount);

    if (!ciDeviceCount) {
        printf("No opencl specific devices!\n");
        return -1;
    }

    printf("Creating Command Queue...\n");
    // create a commiand queuw on device 1
    createCommandQueue(1);

    printf("Compiling Program..\n");
    cl_program cpProgram;
    if (compileProgram(argv, "fft.h", "kernels/slowfft.cl", 1, cpProgram) != 
                                                                  CL_SUCCESS) {
        printf("Compilation failed.\n");
	printCompilationErrors(cpProgram, 1);
        return -1;
    }
  
    cl_kernel kernobj;
    if (createKernel(cpProgram,"slowfft", kernobj) != CL_SUCCESS) {
        printf("Kernel Creation failure.\n");
        return -1;
    }
  
    // Allocate host memory. 
    // h_Freal and h_Fimag represent the input signal to be transformed.
    // h_Rreal and h_Rimag represent the transformed output.

    float* const h_Freal = (float *) malloc(sizeof(float) * ARR_SIZE);
    if (h_Freal == NULL) {
        printf("Error Could not allocate memory!\n");
        return -1;
    }
    float* const h_Fimag = (float *) malloc(sizeof(float) * ARR_SIZE);
    if (h_Fimag == NULL) {
        printf("Error could not allocate memory!\n");
        return -1;
    }
    float* const h_Rreal = (float *) malloc(sizeof(float) * ARR_SIZE);
    if (h_Rreal == NULL) {
        printf("Error could not allocate memory!\n");
        return -1;
    }
    float* const h_Rimag = (float *) malloc(sizeof(float) * ARR_SIZE);
    if (h_Rimag == NULL) {
        printf("Error could not allocate memory!\n");
        return -1;
    }
    

    printf("Initializing Arrays.. \n");
    for (unsigned i = 0 ; i < ARR_SIZE; ++i) {
        h_Freal[i] = 1.0;
        h_Fimag[i] = 1.0;
        h_Rreal[i] = 0.0;
	h_Rimag[i] = 0.0;
    }


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

  
    clSetKernelArg(kernobj, 0, sizeof(cl_mem), (void*) &d_Freal);
    clSetKernelArg(kernobj, 1, sizeof(cl_mem), (void*) &d_Fimag);
    clSetKernelArg(kernobj, 2, sizeof(cl_mem), (void*) &d_Rreal);
    clSetKernelArg(kernobj, 3, sizeof(cl_mem), (void*) &d_Rimag);
    clSetKernelArg(kernobj, 4, sizeof(int), (void*)n);
    clSetKernelArg(kernobj, 5, sizeof(int), (void*)is);
    


    runKernel(kernobj, 1, /* Work Dimension. */
       	      localWorkSize,
	      globalWorkSize);

    copyFromDevice(d_Rreal, sizeof(float) * ARR_SIZE, h_Rreal, true);
    copyFromDevice(d_Rimag, sizeof(float) * ARR_SIZE, h_Rimag, true);

    printf("Results : \n");
    for (unsigned i = 0; i <ARR_SIZE; ++i) 
      {
	printf("%f + i%f ", h_Rreal[i], h_Rimag); 
      }
}
