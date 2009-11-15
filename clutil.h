/* 
   File: clutil.h
 */

#ifndef __CLUTIL__
#define __CLUTIL__

#include <oclUtils.h>

cl_int init_cl_context(cl_device_type device_type);

cl_int getDeviceCount(cl_uint * ciDeviceCount);

cl_int createCommandQueue(cl_device_id * device, int device_id);

cl_int compile_program(char * argv[] , char * header_file, 
		       char * kernel_file,
		       cl_program * cpProgram); /* Program object stored in here. */


cl_int  createKernel(cl_program cpProgram, 
		     char * kernelName,
		     cl_kernel *kernobj);

cl_mem createDeviceBuffer(cl_mem_flags flags, size_t size,void * host_ptr,
			  int copytoDevice);

cl_int runKernel(cl_kernel kernobj, cl_uint workDim, size_t * localWorkSize,
		 size_t * globalWorkSize);

void copyfromDevice(cl_mem d_Mem, size_t size, void * h_Mem, cl_int deviceCount);
#endif
