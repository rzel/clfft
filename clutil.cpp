/* 
 File: clutil.c 
 Functions to help with opencl management.
*/
#include "clutil.h"
using namespace std;
// global variables
cl_context cxContext = 0;
cl_program cpProgram = 0;
cl_kernel kernel[MAX_GPU_COUNT];
cl_command_queue commandQueue[MAX_GPU_COUNT];
cl_event gpuExecution[MAX_GPU_COUNT];
cl_event gpuDone[MAX_GPU_COUNT];

// Default Configs
unsigned useCpu = 0;
unsigned deviceCount = 1;
unsigned blockSize = 16;
unsigned fftAlgo = 1;
unsigned print = 0;
// host memory
// h_Freal and h_Fimag represent the input signal to be transformed.
// h_Rreal and h_Rimag represent the transformed output.
float*  h_Freal = 0;
float*  h_Fimag = 0;
float*  h_Rreal = 0;
float*  h_Rimag = 0;

// device memory
// d_Freal and d_Fimag represent the input signal to be transformed.
// d_Rreal and d_Rimag represent the transformed output.
cl_mem d_Freal[MAX_GPU_COUNT];
cl_mem d_Fimag[MAX_GPU_COUNT];
cl_mem d_Rreal[MAX_GPU_COUNT];
cl_mem d_Rimag[MAX_GPU_COUNT];

unsigned
initExecution(const unsigned size, const unsigned samplesize)
{
    // Allocate host memory
  allocateHostMemory(size, samplesize);
    if (deviceCount) {
        cout << "Initializing device(s).." << endl;
        // create the OpenCL context on available GPU devices
        init_cl_context(CL_DEVICE_TYPE_GPU);

        const cl_uint ciDeviceCount =  getDeviceCount();


        if (!ciDeviceCount) {
            printf("No opencl specific devices!\n");
            return 0;
        }

        printf("Creating Command Queue...\n");
        // create a command queue on device 1
        for (unsigned i = 0; i < deviceCount; ++i) {
            createCommandQueue(i);
        }
    }
    return 1;
}

void 
partition(const unsigned size, unsigned& sizeOnGPU, unsigned& sizeOnCPU)
{
    if (deviceCount == 0) {
        sizeOnGPU = 0;
        sizeOnCPU = size;
        return;
    }
   
    if (useCpu == 0) {
       sizeOnGPU = size;
       sizeOnCPU = 0;
       return;
    }    

    sizeOnGPU = (size / 2);
    sizeOnCPU = size - sizeOnGPU; 
}

void 
printGpuTime()
{
    for (unsigned k = 0; k < deviceCount; ++k) {
        cout << "Kernel execution time on GPU " << k
             << " :  " << executionTime(k) << " seconds" << endl;
    }
}

void
printCpuTime(struct rusage& start, struct rusage& end)
{
    const time_t startSec = start.ru_stime.tv_sec;
    const suseconds_t startUsec = start.ru_stime.tv_usec;

    const time_t endSec = end.ru_stime.tv_sec;
    const suseconds_t endUsec = end.ru_stime.tv_usec;
    
    const time_t diffSec =  endSec - startSec;
    const suseconds_t diffUsec =  endUsec - startUsec;
     
    cout << "CPU Time Taken " 
         << diffSec << " seconds " 
         << diffUsec << " micro seconds"
         << endl;

}

void
printResult(const unsigned size)
{
     for (unsigned i = 0; i < size; ++i) {
         printf("%f + i%f \n", h_Rreal[i], h_Rimag[i]);
     }
}

double 
executionTime(const unsigned device)
{
    cl_ulong start, end;
    clGetEventProfilingInfo(gpuExecution[device], 
                            CL_PROFILING_COMMAND_END, 
                            sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(gpuExecution[device],
                            CL_PROFILING_COMMAND_START, 
                            sizeof(cl_ulong), &start, NULL);

    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds 
}

void
allocateHostMemory(const unsigned size, const unsigned n)
{
    h_Freal = (float *) malloc(sizeof(float) * size);
    checkError((h_Freal != NULL), shrTRUE, "Could not allocate memory");

    h_Fimag = (float *) malloc(sizeof(float) * size);
    checkError((h_Fimag != NULL), shrTRUE, "Could not allocate memory");

    h_Rreal = (float *) malloc(sizeof(float) * size);
    checkError((h_Rreal != NULL), shrTRUE, "Could not allocate memory");

    h_Rimag = (float *) malloc(sizeof(float) * size);
    checkError((h_Rimag != NULL), shrTRUE, "Could not allocate memory");

    for (unsigned i = 0 ; i < size; ++i) {
        h_Freal[i] = (i + 1)%n;
        h_Fimag[i] = (i + 1)%n;
        h_Rreal[i] = 0;
        h_Rimag[i] = 0;
    }

}

void
allocateDeviceMemory(const unsigned device, const unsigned size, 
                                      const unsigned copyOffset)
{
    d_Freal[device] = createDeviceBuffer(
                        CL_MEM_READ_ONLY,
                        sizeof(float) * size,
                        h_Freal + copyOffset);
    copyToDevice(device, d_Freal[device],  h_Freal + copyOffset, size);

    d_Fimag[device] = createDeviceBuffer(
                        CL_MEM_READ_ONLY,
                        sizeof(float) * size,
                        h_Fimag + copyOffset);
    copyToDevice(device, d_Fimag[device],  h_Fimag + copyOffset, size);

    d_Rreal[device] = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Rreal + copyOffset);
    copyToDevice(device, d_Rreal[device],  h_Rreal + copyOffset, size);

    d_Rimag[device] = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Rimag + copyOffset);
    copyToDevice(device, d_Rimag[device],  h_Rimag + copyOffset, size);
}

void 
cleanup()
{
    for (unsigned i = 0; i < deviceCount; ++i) { 
        // clean up device
        if (d_Freal[i]) { 
            clReleaseMemObject(d_Freal[i]);
        }
        if (d_Fimag[i]) { 
            clReleaseMemObject(d_Fimag[i]);
        }
        if (d_Rreal[i]) {
            clReleaseMemObject(d_Rreal[i]);
        }
        if (d_Rimag[i]) { 
            clReleaseMemObject(d_Rimag[i]);
        }
        // cleanup ocl routines
        if (gpuExecution[i]) {  
            clReleaseEvent(gpuExecution[i]);
        }
        if (gpuDone[i]) {
            clReleaseEvent(gpuDone[i]);
        }
        if (kernel[i]) { 
            clReleaseKernel(kernel[i]);
        }
        if (commandQueue[i]) {
            clReleaseCommandQueue(commandQueue[i]);
        }
    }

    if (cpProgram) {
        clReleaseProgram(cpProgram);
    }
    if (cxContext) { 
        // segfaults if context is null.
        clReleaseContext(cxContext);
    }

    // Release mem and event objects 
    free(h_Freal);
    h_Freal = 0;
    free(h_Fimag);
    h_Fimag = 0;
    free(h_Rreal);
    h_Rreal = 0;
    free(h_Rimag);
    h_Rimag = 0;
}

void
checkError(const cl_int ciErrNum, const cl_int ref, 
                      const char* const operation) 
{
    if (ciErrNum != ref) {
        printf("ERROR:: %d %s failed\n\n", ciErrNum, operation); 
        cleanup();
        // TODO::        may be print the type of error
        exit(EXIT_FAILURE);
    }    
}

void
init_cl_context(const cl_device_type device_type)
{
    cl_int ciErrNum = CL_SUCCESS; 

    cxContext = clCreateContextFromType(0, /* cl_context_properties */
  				        device_type,
				        NULL, /* error function ptr */
				        NULL, /* user data to be passed to err fn */
				        &ciErrNum);

    checkError(ciErrNum, CL_SUCCESS, "clCreateContextFromType");
}

cl_uint 
getDeviceCount()
{
    size_t nDeviceBytes;
  
    const cl_int ciErrNum = clGetContextInfo(cxContext,
	          	      CL_CONTEXT_DEVICES, /* Param Name */
			      0, /* size_t param_value_size  */
			      NULL, /* void * param_value */
			      &nDeviceBytes); /* size_t param_value_size_ret */
  
    checkError(ciErrNum, CL_SUCCESS, "clGetContextInfo");
    return ((cl_uint)nDeviceBytes/sizeof(cl_device_id));
}
      

void
createCommandQueue(const unsigned deviceId)
{
    const cl_device_id device = oclGetDev(cxContext, deviceId);
    cl_int ciErrNum = CL_SUCCESS;
    commandQueue[deviceId]  = clCreateCommandQueue(cxContext, 
			               	           device,
				                   0,
				                   &ciErrNum);
    checkError(ciErrNum, CL_SUCCESS, "clCreateCommandQueue"); 
    ciErrNum = clSetCommandQueueProperty(commandQueue[deviceId], 
                                         CL_QUEUE_PROFILING_ENABLE, 
                                         CL_TRUE, NULL);
    checkError(ciErrNum, CL_SUCCESS, "clSetCommandQueueProperty");   
}				      

void
compileProgram(const char* const argv[] , const char* const header_file, 
               const char* const kernel_file)
{

    // Load the OpenCL source code from the .cl file 
    const char* header_path = shrFindFilePath(header_file, argv[0]);
  
    size_t program_length;
    char* header = oclLoadProgSource(header_path, "" , &program_length);

    checkError((header != NULL), shrTRUE, "oclLoadProgSource on header");

    const char* source_path = shrFindFilePath(kernel_file, argv[0]);
    char* source = oclLoadProgSource(source_path, header, &program_length);

    checkError((source != NULL), shrTRUE, "oclLoadProgSource on source");

    cl_int ciErrNum; 
    // Create the program for all GPUs in the context
    cpProgram = clCreateProgramWithSource( cxContext, 1,
		 			   (const char **) &source,
					   &program_length,
					   &ciErrNum);
    free(header);
    free(source);
     
    checkError(ciErrNum, CL_SUCCESS, "clCreateProgramWithSource"); 
    /* Build program */

    ciErrNum = clBuildProgram(cpProgram,
        		      0, 	/* Number of devices for which we need to do this */
			      NULL, /* Device List */
			      "",
			      NULL, /* ptr to function */
			      NULL); /* User data to pass to ptrfn */

    if (ciErrNum != CL_SUCCESS) {
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxContext));
        checkError(ciErrNum, CL_SUCCESS, "clBuildProgram");
    }
}

void
createKernel(const unsigned device, const char* const kernelName)
{
    cl_int ciErrNum = CL_SUCCESS;
    kernel[device] = clCreateKernel(cpProgram, kernelName, &ciErrNum);
    checkError(ciErrNum, CL_SUCCESS, "clCreateKernel");
}


cl_mem 
createDeviceBuffer(const cl_mem_flags flags, 
                   const size_t size, void* const hostPtr)
	           


{
    cl_int ciErrNum = CL_SUCCESS;
    const cl_mem d_mem = clCreateBuffer(cxContext,
 	 	                        flags,
		                        size,
		                        hostPtr,
		                        &ciErrNum);
  
    checkError(ciErrNum, CL_SUCCESS,  "clCreateBuffer");
    return d_mem;
}

void 
copyToDevice(const unsigned device, const cl_mem mem, 
             float* const hostPtr, const unsigned size)
{
    const cl_int ciErrNum = clEnqueueWriteBuffer(commandQueue[device], 
                                                mem, CL_TRUE, 0, 
                                                sizeof(float) * size, hostPtr, 
                                                0, NULL, NULL);
    checkError(ciErrNum, CL_SUCCESS,  "clEnqueueWriteBuffer");
}

void
copyFromDevice(const unsigned device, const cl_mem dMem,
               float* const hostPtr, const unsigned size)
{
    cl_int ciErrNum = clEnqueueReadBuffer(commandQueue[device], dMem,
                                          CL_FALSE, 0,
                                          sizeof(float) * size,
                                          hostPtr, 0, NULL, 
                                          &gpuDone[device]);
    checkError(ciErrNum, CL_SUCCESS, "clEnqueueReadBuffer"); 
}


void 
runKernel(const unsigned device, const size_t localWorkSize[], 
                                 const size_t globalWorkSize[])
{

    const cl_int ciErrNum = clEnqueueNDRangeKernel(commandQueue[device], 
                                                   kernel[device], 
                                                   1, NULL, 
                                	           globalWorkSize, 
                                                   localWorkSize, 0, 
                                                   NULL, 
                                                   &gpuExecution[device]);
    checkError(ciErrNum, CL_SUCCESS, "clEnqueueNDRangeKernel");
}

