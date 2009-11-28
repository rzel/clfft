
#include "clutil.h"
#include "fft.h"
#include "kernels.h"


int 
main(const int argc, const char* argv[])
{
    
    if (argc < 2) {
        printf("Usage: clfft N [i]\n");
	return 0;
    }

    const unsigned n = atoi(argv[1]);
    int is = -1;
    if (argc == 3) {		// This is a weak check, but should do for now.
        is = 1;
    }

    // Allocate host memory. 
    allocateHostMemory(ARR_SIZE);

    printf("Initializing Arrays.. \n");
    for (unsigned i = 0 ; i < ARR_SIZE; ++i) {
        h_Freal[i] = i + 1;
        h_Fimag[i] = i + 1;
        h_Rreal[i] = 0.0;
	h_Rimag[i] = 0.0;
    }

    printf("Initializing CL Context..\n");
    // create the OpenCL context on available GPU devices
    init_cl_context(CL_DEVICE_TYPE_GPU);

    printf("Getting Device Count..\n");
    const cl_uint ciDeviceCount =  getDeviceCount();


    if (!ciDeviceCount) {
        printf("No opencl specific devices!\n");
        return 0;
    }
    deviceCount = 1;
    printf("Creating Command Queue...\n");
    // create a command queuw on device 1
    for (unsigned i = 0; i < deviceCount; ++i) {
        createCommandQueue(i);
    }

    //cooleyTukey(argv, n, ARR_SIZE);  
    //slowFFT(argv, n, is, ARR_SIZE);

    stockhamFFT(argv, n, is, ARR_SIZE);

    for (unsigned i = 0; i < deviceCount; ++i) {
        printf("Kernel execution time on GPU %d: %.9f s\n", i, executionTime(i));
    }
    cleanup();
    return 1;
}

