/* 
 File: clutil.c 
 Functions to help with opencl management.
*/
#include "clutil.h"

// global variables
cl_context cxContext = 0;
//TODO:: to be made an array of MAX_GPU
cl_kernel kernel = 0;
cl_command_queue commandQueue = 0;
cl_event event = 0;
cl_program cpProgram = 0;


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
cl_mem d_Freal = 0;
cl_mem d_Fimag = 0;
cl_mem d_Rreal = 0;
cl_mem d_Rimag = 0;

double 
executionTime()
{
    cl_ulong start, end;
    assert(event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
                            sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
                            sizeof(cl_ulong), &start, NULL);

    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds 
}

void
allocateHostMemory(const unsigned size)
{
    h_Freal = (float *) malloc(sizeof(float) * size);
    checkError((h_Freal != NULL), shrTRUE, "Could not allocate memory");

    h_Fimag = (float *) malloc(sizeof(float) * size);
    checkError((h_Fimag != NULL), shrTRUE, "Could not allocate memory");

    h_Rreal = (float *) malloc(sizeof(float) * size);
    checkError((h_Rreal != NULL), shrTRUE, "Could not allocate memory");

    h_Rimag = (float *) malloc(sizeof(float) * size);
    checkError((h_Rimag != NULL), shrTRUE, "Could not allocate memory");
}

void
allocateDeviceMemory(const unsigned size)
{
    d_Freal = createDeviceBuffer(
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * size,
                        h_Freal,
                        true);

    d_Fimag = createDeviceBuffer(
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * size,
                        h_Fimag,
                        true);

    d_Rreal = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Rreal,
                                              true);

    d_Rimag = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Rimag,
                                              true);
}

void 
cleanup()
{
    // clean up device
    clReleaseMemObject(d_Freal);
    clReleaseMemObject(d_Fimag);
    clReleaseMemObject(d_Rreal);
    clReleaseMemObject(d_Rimag);

    // cleanup ocl routines
    clReleaseEvent(event);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(cpProgram);
    clReleaseContext(cxContext);

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
checkError(const cl_int ciErrNum, const cl_int ref, const char* const operation) 
{
    if (ciErrNum != ref) {
        printf("ERROR:: %d %s failed\n\n", ciErrNum, operation); 
        cleanup();
        // TODO:: cleanup the memories
        //        may be print the type of error
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

    fprintf(stdout,"After createContext ..\n");
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
    commandQueue = clCreateCommandQueue(cxContext, 
				        device,
				        0,
				        &ciErrNum);
    checkError(ciErrNum, CL_SUCCESS, "clCreateCommandQueue"); 
    ciErrNum = clSetCommandQueueProperty(commandQueue, 
                                         CL_QUEUE_PROFILING_ENABLE, 
                                         CL_TRUE, NULL);
   
    checkError(ciErrNum, CL_SUCCESS, "clSetCommandQueueProperty");   
}				      

void
compileProgram(const char* const argv[] , const char* const header_file, 
               const char* const kernel_file,const unsigned deviceid)
{
    const char* header_path = shrFindFilePath(header_file, argv[0]);
  
    size_t program_length;
    char* header = oclLoadProgSource(header_path, "" , &program_length);

    checkError((header != NULL), shrTRUE, "oclLoadProgSource on header");

    const char* source_path = shrFindFilePath(kernel_file, argv[0]);
    char* source = oclLoadProgSource(source_path, header, &program_length);

    checkError((source != NULL), shrTRUE, "oclLoadProgSource on source");

    cl_int ciErrNum;
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
printCompilationErrors(const cl_program& cpProgram, const unsigned deviceId)
{
    const cl_device_id device = oclGetDev(cxContext, deviceId);

    size_t len;
    char buffer[4096];
    const cl_int ciErrNum = clGetProgramBuildInfo(cpProgram, device, 
                                                  CL_PROGRAM_BUILD_LOG,
                                                  sizeof(buffer), buffer, &len);    
    
    if (ciErrNum != CL_SUCCESS) {
        switch(ciErrNum) {
	    case CL_INVALID_DEVICE: 
            fprintf(stderr,"Unable to get Build info: INVALID_DEVICE.\n"); 
            break;
	    case CL_INVALID_VALUE: 
            fprintf(stderr,"Unable to get Build info: INVALID_VALUE.\n"); 
            break;
	    case CL_INVALID_PROGRAM: 
            fprintf(stderr,"Unable to get Build info: INVALID_PROGRAM.\n"); 
            break;
	    default: 
            fprintf(stderr,"Unable to get Build info.\n"); 
            break;
	 }
    }

    fprintf(stderr,"Error %d:%s\n",ciErrNum, buffer);
  
}


void
createKernel(const char* const kernelName)
{
    cl_int ciErrNum = CL_SUCCESS;
    kernel = clCreateKernel(cpProgram, kernelName, &ciErrNum);
    checkError(ciErrNum, CL_SUCCESS, "clCreateKernel");
}


cl_mem 
createDeviceBuffer(const cl_mem_flags flags, 
                   const size_t size, void* const hostPtr,
	           const bool copyToDevice)


{
    cl_int ciErrNum = CL_SUCCESS;
    const cl_mem d_mem = clCreateBuffer(cxContext,
 	 	                        flags,
		                        size,
		                        hostPtr,
		                        &ciErrNum);
  
    checkError(ciErrNum, CL_SUCCESS,  "clCreateBuffer");
    if (copyToDevice) {
        fprintf(stderr,"Copyting to device memory.\n");
        ciErrNum = clEnqueueWriteBuffer(commandQueue, d_mem, CL_TRUE, 0, size,
			                hostPtr, 0, NULL, NULL);
        checkError(ciErrNum, CL_SUCCESS,  "clEnqueueWriteBuffer"); 
    }

    return d_mem;
}



void 
runKernel(const cl_kernel kernobj, const cl_uint workDim, 
          const size_t localWorkSize[], const size_t globalWorkSize[])
{

    event = NULL;
    const cl_int ciErrNum = clEnqueueNDRangeKernel(commandQueue, kernobj, 
                                                   workDim, NULL, 
                                	           globalWorkSize, 
                                                   localWorkSize, 0, 
                                                   NULL, &event);
    checkError(ciErrNum, CL_SUCCESS, "clEnqueueNDRangeKernel");
}

void 
copyFromDevice(const cl_mem dMem, const size_t size, 
               void* const hMem, const cl_int deviceCount)
{
    cl_event GPUDone;
    cl_int ciErrNum = clEnqueueReadBuffer(commandQueue, dMem,CL_TRUE, 0, size,
  		                                hMem, 0, NULL, &GPUDone);
    checkError(ciErrNum, CL_SUCCESS, "clEnqueueReadBuffer"); 

    //TODO:: we dont need to wait before copying real and imag
    ciErrNum = clWaitForEvents(deviceCount, &GPUDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");
}
