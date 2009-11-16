/* 
 File: clutil.c 
 Functions to help with opencl management.
*/
#include "clutil.h"

cl_context cxContext = NULL;
cl_command_queue commandQueue;



cl_int init_cl_context(cl_device_type device_type)
{
  cl_int ciErrNum = CL_SUCCESS; 

  cxContext = clCreateContextFromType(0, /* cl_context_properties */
				      device_type,
				      NULL, /* error function ptr */
				      NULL, /* user data to be passed to err fn */
				      &ciErrNum);

  fprintf(stdout,"After createContext ..\n");

  if(ciErrNum != CL_SUCCESS)
    {
      fprintf(stderr,"Error: Context Creation Failed:%d!\n", ciErrNum);
    }

  return ciErrNum;
}

cl_int getDeviceCount(cl_uint * ciDeviceCount)
{
  cl_int ciErrNum = CL_SUCCESS;
  size_t nDeviceBytes;
  
  ciErrNum = clGetContextInfo(cxContext,
			      CL_CONTEXT_DEVICES, /* Param Name */
			      0, /* size_t param_value_size  */
			      NULL, /* void * param_value */
			      &nDeviceBytes); /* size_t param_value_size_ret */
  
  if(ciErrNum != CL_SUCCESS)
    {
      fprintf(stderr,"Error in GetDeviceInfo!\n");
      return ciErrNum;
    }

  *ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

  if(*ciDeviceCount == 0)
    fprintf(stderr, "No Devices Supporting OpenCL!!\n");

  return ciErrNum;
}
      

cl_int createCommandQueue(cl_device_id * device , int device_id)
{
  cl_int ciErrNum = CL_SUCCESS;

  *device = oclGetDev(cxContext, device_id);
  commandQueue = clCreateCommandQueue( cxContext, 
				       *device,
				       0,
				       &ciErrNum);

  if(ciErrNum != CL_SUCCESS)
    {
      fprintf(stderr, "Could not create CommandQueue !!\n");
    }

  return ciErrNum;
}				      

cl_int compile_program(char * argv[] , char * header_file, 
		       char * kernel_file,
		       cl_program * cpProgram) /* Program object stored in here. */
{
  cl_int ciErrNum;
  size_t program_length;
  const char * header_path = shrFindFilePath(header_file, argv[0]);
  
  char * header = oclLoadProgSource( header_path, "" , &program_length);
  char * source_path, * source;

  if(header == NULL)
    {
      fprintf(stderr,"Error: Failed to load the header %s!\n", header_path);
      return -1000;
    }

  source_path = shrFindFilePath(kernel_file, argv[0]);
  source = oclLoadProgSource(source_path, header, &program_length);

  if(!source)
    {
      fprintf(stderr, "Error: Failed to load source %s !\n", source);
      return -2000;
    }

  *cpProgram = clCreateProgramWithSource(cxContext, 
					1,
					(const char **) &source,
					&program_length,
					&ciErrNum);

  free(header);
  free(source);
  if(ciErrNum != CL_SUCCESS)
    {
      fprintf(stderr, "Error: Failed to create Program!\n");
      return ciErrNum;
    }
      
  /* Build program */

  ciErrNum = clBuildProgram(*cpProgram,
			    0, 	/* Number of devices for which we need to do this */
			    NULL, /* Device List */
			    "-cl-mad-enable",
			    NULL, /* ptr to function */
			    NULL); /* User data to pass to ptrfn */

  if(ciErrNum != CL_SUCCESS)
    {
      fprintf(stderr,"Error: Compilation Failure! \n");
    }

  return ciErrNum;
}

cl_int  createKernel(cl_program cpProgram, 
		     char * kernelName,
		     cl_kernel *kernobj)
{

  cl_int ciErrNum = CL_SUCCESS;

  *kernobj = clCreateKernel(cpProgram, kernelName, &ciErrNum);

  if(ciErrNum != CL_SUCCESS)
    fprintf(stderr, "Error: Failed to Create Kernel: %s!\n", kernelName);
      
  return ciErrNum;
}


cl_mem createDeviceBuffer(cl_mem_flags flags, size_t size,void * host_ptr,
			  int copytoDevice)
{
  cl_int ciErrNum = CL_SUCCESS;
  cl_mem d_mem;
  d_mem = clCreateBuffer(cxContext,
		 flags,
		 size,
		 host_ptr,
		 &ciErrNum);
  
  if(ciErrNum != CL_SUCCESS)
    {
      fprintf(stderr,"Couldnt create device buffer !!");
      return NULL;
    }

  if(copytoDevice)
    {
      fprintf(stderr,"Copyting to device memory.\n");
      clEnqueueWriteBuffer(commandQueue, d_mem, CL_TRUE, 0, size,
			   host_ptr, 0, NULL, NULL);
    }

  return d_mem;
}



cl_int runKernel(cl_kernel kernobj, cl_uint workDim, size_t localWorkSize[],
		 size_t  globalWorkSize[])
{

  cl_event GPUExecution;
  cl_int ciErrNum;

  fprintf(stderr,"Before Kernel .. kernobj:%x globalWorkSize:%x localWorkSize:%x \n", kernobj, *globalWorkSize, *localWorkSize);
  ciErrNum = clEnqueueNDRangeKernel(commandQueue, kernobj, workDim, NULL, 
			 globalWorkSize, localWorkSize, 0, NULL,
			 NULL);
  fprintf(stderr,"After kernel..\n");

  if(ciErrNum != CL_SUCCESS)
    {

      fprintf(stderr,"Kernel Enqueue not a success !!\n");

      switch(ciErrNum)
	{
	case CL_INVALID_WORK_GROUP_SIZE: fprintf(stderr,"CL_INVALID_WORK_GROUP_SIZE\n"); break;
	case CL_INVALID_WORK_ITEM_SIZE: fprintf(stderr,"CL_INVALID_WORK_GROUP_SIZE\n"); break;
	case CL_INVALID_WORK_DIMENSION: fprintf(stderr,"CL_INVALID_WORK_DIMENSION\n"); break;  
	case CL_INVALID_KERNEL: fprintf(stderr,"CL_INVALID_KERNEL\n"); break;
	case CL_INVALID_KERNEL_ARGS: fprintf(stderr,"CL_INVALID_KERNEL_ARGS\n"); break;
	case CL_INVALID_VALUE : fprintf(stderr,"CL_INVALID_VALUE\n"); break;
	default : fprintf(stderr,"Unknown case %d \n",ciErrNum);
	}
      
    }
  
  return ciErrNum;
}

void copyfromDevice(cl_mem d_Mem, size_t size, void * h_Mem, cl_int deviceCount)
{
  cl_event GPUDone;
  clEnqueueReadBuffer(commandQueue, d_Mem,CL_TRUE, 0, size,
		      h_Mem, 0, NULL, &GPUDone);

  clWaitForEvents(deviceCount, &GPUDone);
  
}
