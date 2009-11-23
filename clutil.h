/* 
   File: clutil.h
 */

#ifndef __CLUTIL__
#define __CLUTIL__

#include <oclUtils.h>

void checkError(const cl_int ciErrNum, const cl_int ref, const char* const operation);


void init_cl_context(cl_device_type device_type);

cl_uint getDeviceCount();

void createCommandQueue(const unsigned deviceId);

void compileProgram(const char* const argv[] , const char* const header_file, 
                    const char* const kernel_file, const unsigned deviceid);


void createKernel(const char* const kernelName);

cl_mem createDeviceBuffer(const cl_mem_flags flags, const size_t size,
                          void* const  hostPtr, const bool copytoDevice);

void runKernel(const cl_kernel  kernobj, const cl_uint workDim, 
               const size_t* localWorkSize, const  size_t* globalWorkSize);

void copyFromDevice(const cl_mem dMem, const size_t size, 
                    void* hMem, const cl_int deviceCount);

void printCompilationErrors(const cl_program& cpProgram, const unsigned deviceId);
double executionTime();
void allocateHostMemory(const unsigned size);
void allocateDeviceMemory(const unsigned size);
void cleanup();

extern float*  h_Freal;
extern float*  h_Fimag;
extern float*  h_Rreal;
extern float*  h_Rimag;

extern cl_mem d_Freal;
extern cl_mem d_Fimag;
extern cl_mem d_Rreal;
extern cl_mem d_Rimag;


extern cl_kernel kernel;
extern cl_event event;
extern cl_program cpProgram;
extern cl_command_queue commandQueue;
extern cl_context cxContext ;
#endif
