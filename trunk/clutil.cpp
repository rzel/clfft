/* 
 File: clutil.c 
 Functions to help with opencl management.
*/
#include "clutil.h"

cl_context cxContext = NULL;
cl_command_queue commandQueue;



cl_int 
init_cl_context(const cl_device_type device_type)
{
    cl_int ciErrNum = CL_SUCCESS; 

    cxContext = clCreateContextFromType(0, /* cl_context_properties */
  				        device_type,
				        NULL, /* error function ptr */
				        NULL, /* user data to be passed to err fn */
				        &ciErrNum);

    fprintf(stdout,"After createContext ..\n");

    if(ciErrNum != CL_SUCCESS) {
        fprintf(stderr,"Error: Context Creation Failed:%d!\n", ciErrNum);
    }

    return ciErrNum;
}

cl_int 
getDeviceCount(cl_uint& ciDeviceCount)
{
    size_t nDeviceBytes;
  
    const cl_int ciErrNum = clGetContextInfo(cxContext,
	          	      CL_CONTEXT_DEVICES, /* Param Name */
			      0, /* size_t param_value_size  */
			      NULL, /* void * param_value */
			      &nDeviceBytes); /* size_t param_value_size_ret */
  
    if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr,"Error in GetDeviceInfo!\n");
        return ciErrNum;
    }

    ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

    if (ciDeviceCount == 0) {
        fprintf(stderr, "No Devices Supporting OpenCL!!\n");
    }

    return ciErrNum;
}
      

cl_int 
createCommandQueue(const unsigned deviceId)
{
    const cl_device_id device = oclGetDev(cxContext, deviceId);
    cl_int ciErrNum = CL_SUCCESS;
    commandQueue = clCreateCommandQueue(cxContext, 
				        device,
				        0,
				        &ciErrNum);

    if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr, "Could not create CommandQueue !!\n");
    }

    return ciErrNum;
}				      

cl_int 
compileProgram(const char* const argv[] , const char* const header_file, 
               const char* const kernel_file,
               cl_program& cpProgram) /* Program object stored in here. */
{
    const char* header_path = shrFindFilePath(header_file, argv[0]);
  
    size_t program_length;
    char* header = oclLoadProgSource(header_path, "" , &program_length);

    if(header == NULL) {
        fprintf(stderr,"Error: Failed to load the header %s!\n", header_path);
        return -1000;
    }

    const char* source_path = shrFindFilePath(kernel_file, argv[0]);
    char* source = oclLoadProgSource(source_path, header, &program_length);

    if (!source) {
        fprintf(stderr, "Error: Failed to load source %s !\n", source);
        return -2000;
    }

    cl_int ciErrNum;
    cpProgram = clCreateProgramWithSource(cxContext, 1,
					   (const char **) &source,
					   &program_length,
					   &ciErrNum);
    // do we need these frees?
    free(header);
    free(source);
    if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create Program!\n");
        return ciErrNum;
    }
      
    /* Build program */

    ciErrNum = clBuildProgram(cpProgram,
        		      0, 	/* Number of devices for which we need to do this */
			      NULL, /* Device List */
			      "-cl-mad-enable",
			      NULL, /* ptr to function */
			      NULL); /* User data to pass to ptrfn */

    if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr,"Error: Compilation Failure! \n");
    }

    return ciErrNum;
}

cl_int  
createKernel(const cl_program& cpProgram, 
	     const char* const kernelName,
	     cl_kernel& kernobj)
{

    cl_int ciErrNum = CL_SUCCESS;
    kernobj = clCreateKernel(cpProgram, kernelName, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to Create Kernel: %s!\n", kernelName);
    }
    return ciErrNum;
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
  
    if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr,"Couldnt create device buffer !!");
        return NULL;
    }

    if (copyToDevice) {
      fprintf(stderr,"Copyting to device memory.\n");
      clEnqueueWriteBuffer(commandQueue, d_mem, CL_TRUE, 0, size,
			   hostPtr, 0, NULL, NULL);
    }

    return d_mem;
}



cl_int 
runKernel(const cl_kernel kernobj, const cl_uint workDim, 
          const size_t localWorkSize[], const size_t globalWorkSize[])
{

    cl_event GPUExecution;

    const cl_int ciErrNum = clEnqueueNDRangeKernel(commandQueue, kernobj, 
                                                   workDim, NULL, 
                                	           globalWorkSize, 
                                                   localWorkSize, 0, 
                                                   NULL, NULL);
    fprintf(stderr,"After kernel..\n");

    if (ciErrNum != CL_SUCCESS) {

        fprintf(stderr,"Kernel Enqueue not a success !!\n");

        switch(ciErrNum) {
            case CL_INVALID_WORK_GROUP_SIZE: 
            fprintf(stderr,"CL_INVALID_WORK_GROUP_SIZE\n"); 
            break;
	    case CL_INVALID_WORK_ITEM_SIZE: 
            fprintf(stderr,"CL_INVALID_WORK_GROUP_SIZE\n"); 
            break;
	    case CL_INVALID_WORK_DIMENSION: 
            fprintf(stderr,"CL_INVALID_WORK_DIMENSION\n"); 
            break;  
	    case CL_INVALID_KERNEL: 
            fprintf(stderr,"CL_INVALID_KERNEL\n"); 
            break;
	    case CL_INVALID_KERNEL_ARGS: 
            fprintf(stderr,"CL_INVALID_KERNEL_ARGS\n"); 
            break;
	    case CL_INVALID_VALUE: 
            fprintf(stderr,"CL_INVALID_VALUE\n"); 
            break;
	    default: 
            fprintf(stderr,"Unknown case %d \n",ciErrNum);
        }
      
    }
  
    return ciErrNum;
}

void 
copyFromDevice(const cl_mem dMem, const size_t size, 
               void* const hMem, const cl_int deviceCount)
{
    cl_event GPUDone;
    clEnqueueReadBuffer(commandQueue, dMem,CL_TRUE, 0, size,
  		        hMem, 0, NULL, &GPUDone);

    clWaitForEvents(deviceCount, &GPUDone);
}
