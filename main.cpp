
#include "clutil.h"
#include "fft.h"
#include "kernels.h"


int 
main(const int argc, const char* argv[])
{
    

    if(argc < 2)
      {
	printf("Usage: clfft N [i]\n");
	return 0;
      }

    const unsigned n = atoi(argv[1]);
    int is = -1;
    if (argc == 3) {		// This is a weak check, but should do for now.
        is = 1;
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
        h_Freal[i] = i + 1;
        h_Fimag[i] = i + 1;
        h_Rreal[i] = 0.0;
	h_Rimag[i] = 0.0;
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

    cooleyTukey(argv, h_Freal, h_Fimag, h_Rreal, h_Rimag, n);  
    //slowFFT(argv, h_Freal,  h_Fimag, h_Rreal, h_Rimag, n, is);
    return 1;
}

