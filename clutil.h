/* 
   File: clutil.h
 */

#ifndef __CLUTIL__
#define __CLUTIL__

#include <oclUtils.h>

cl_int init_cl_context(cl_device_type device_type);

cl_int getDeviceCount(cl_uint& ciDeviceCount);

cl_int createCommandQueue(const unsigned deviceId);

cl_int compileProgram(const char* const argv[] , const char* const header_file, 
		       const char* const kernel_file,
		       cl_program& cpProgram); /* Program object stored in here. */


cl_int  createKernel(const cl_program& cpProgram, 
		     const char* const kernelName,
		     cl_kernel& kernobj);

cl_mem createDeviceBuffer(const cl_mem_flags flags, const size_t size,
                          void* const  hostPtr, const bool copytoDevice);

cl_int runKernel(const cl_kernel  kernobj, const cl_uint workDim, 
                 const size_t* localWorkSize, const  size_t* globalWorkSize);

void copyFromDevice(const cl_mem dMem, const size_t size, 
                    void* hMem, const cl_int deviceCount);

void printCompilationErrors(const cl_program& cpProgram, const unsigned deviceId);

extern cl_command_queue commandQueue;
extern cl_context cxContext ;
#endif
