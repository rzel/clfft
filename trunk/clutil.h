/* 
   File: clutil.h
 */

#ifndef __CLUTIL__
#define __CLUTIL__

#include <oclUtils.h>
#include<iostream>
#define MAX_GPU_COUNT 2
#define SLOW_FFT 1
#define COOLEY_TUKEY 2
#define STOCKHALM 3

unsigned initExecution(const unsigned size);
void partition(const unsigned size, unsigned& sizeOnGPU, unsigned& sizeOnCPU);

void checkError(const cl_int ciErrNum, const cl_int ref, const char* const operation);

void printResult(const unsigned size);

void init_cl_context(cl_device_type device_type);

cl_uint getDeviceCount();

void createCommandQueue(const unsigned deviceId);

void compileProgram(const char* const argv[] , const char* const header_file, 
                                               const char* const kernel_file);


void createKernel(const unsigned device, const char* const kernelName);

cl_mem createDeviceBuffer(const cl_mem_flags flags, const size_t size,
                          void* const  hostPtr);

void runKernel(const unsigned device, const size_t* localWorkSize, 
                                    const  size_t* globalWorkSize);

void copyToDevice(const unsigned device, const cl_mem mem,
                  float* const hostPtr, const unsigned size);


void copyFromDevice(const unsigned device, const cl_mem dMem,
                    float* const hostPtr, const unsigned size);

double executionTime(const unsigned device);
void allocateHostMemory(const unsigned size);
void allocateDeviceMemory(const unsigned device, const unsigned size,
                                           const unsigned copyOffset);

void printGpuTime();

void cleanup();

extern unsigned useCpu;
extern unsigned useGpu;
extern unsigned blockSize;
extern unsigned fftAlgo;
extern unsigned print;

extern float*  h_Freal;
extern float*  h_Fimag;
extern float*  h_Rreal;
extern float*  h_Rimag;

extern cl_mem d_Freal[MAX_GPU_COUNT];
extern cl_mem d_Fimag[MAX_GPU_COUNT];
extern cl_mem d_Rreal[MAX_GPU_COUNT];
extern cl_mem d_Rimag[MAX_GPU_COUNT];

extern cl_context cxContext;
extern cl_program cpProgram;
extern cl_event gpuExecution[MAX_GPU_COUNT];
extern cl_event gpuDone[MAX_GPU_COUNT];
extern cl_kernel kernel[MAX_GPU_COUNT];
extern cl_command_queue commandQueue[MAX_GPU_COUNT];

extern unsigned deviceCount;
#endif
