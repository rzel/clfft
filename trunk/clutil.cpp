/* 
 File: clutil.c 
 Functions to help with opencl management.
*/
#include "clutil.h"

cl_context cxContext = NULL;
cl_command_queue commandQueue;

void
checkError(const cl_int ciErrNum, const cl_int ref, const char* const operation) 
{
    if (ciErrNum != ref) {
        shrLog(LOGCONSOLE, ciErrNum, "ERROR:: %s Failed\n\n", operation);
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
}				      

cl_program
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
    const cl_program cpProgram = clCreateProgramWithSource(
                                           cxContext, 1,
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
    return cpProgram;
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


cl_kernel 
createKernel(const cl_program& cpProgram, 
	     const char* const kernelName)
{

    cl_int ciErrNum = CL_SUCCESS;
    const cl_kernel kernobj = clCreateKernel(cpProgram, kernelName, &ciErrNum);
    checkError(ciErrNum, CL_SUCCESS, "clCreateKernel");
    return kernobj;
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

    const cl_int ciErrNum = clEnqueueNDRangeKernel(commandQueue, kernobj, 
                                                   workDim, NULL, 
                                	           globalWorkSize, 
                                                   localWorkSize, 0, 
                                                   NULL, NULL);
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

    ciErrNum = clWaitForEvents(deviceCount, &GPUDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");
}
