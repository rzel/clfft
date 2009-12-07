#include "clutil.h"
#include "fft.h"
#include "kernels.h"

static unsigned workOffset[MAX_GPU_COUNT];
static unsigned workSize[MAX_GPU_COUNT];

//Toeplitz matrix H
float * h_Hreal = 0;
float * h_Himag = 0;

//Y and Z arrays
float * h_Yreal = 0;
float * h_Yimag = 0;

float * h_Zreal = 0;
float * h_Zimag = 0;

//Sample array
float * h_Xreal = 0;
float * h_Ximag = 0;

cl_mem d_Hreal[MAX_GPU_COUNT];
cl_mem d_Himag[MAX_GPU_COUNT];
cl_mem d_Zreal[MAX_GPU_COUNT];
cl_mem d_Zimag[MAX_GPU_COUNT];
cl_mem d_Yreal[MAX_GPU_COUNT];
cl_mem d_Yimag[MAX_GPU_COUNT];


/* 
   Note : Bluesteins will work only where size of input
   array = n . Have yet to implement for size = k* n.
*/


void 
allocateHostMemoryBluesteins(const unsigned n, const unsigned m)
{
    
  h_Hreal = (float *) malloc(sizeof(float) * m);
  checkError((h_Hreal != NULL), shrTRUE, "Could not allocate memory");
    
  h_Himag = (float *) malloc(sizeof(float) * m);
  checkError((h_Himag != NULL), shrTRUE, "Could not allocate memory");
  
  h_Yreal = (float *) malloc(sizeof(float) * m);
  checkError((h_Yreal != NULL), shrTRUE, "Could not allocate memory");
    
  h_Yimag = (float *) malloc(sizeof(float) * m);
  checkError((h_Yimag != NULL), shrTRUE, "Could not allocate memory");
  
  h_Zreal = (float *) malloc(sizeof(float) * m);
  checkError((h_Zreal != NULL), shrTRUE, "Could not allocate memory");
    
  h_Zimag = (float *) malloc(sizeof(float) * m);
  checkError((h_Zimag != NULL), shrTRUE, "Could not allocate memory");
  
  h_Xreal = (float *) malloc(sizeof(float) * m);
  checkError((h_Xreal != NULL), shrTRUE, "Could not allocate memory");
    
  h_Ximag = (float *) malloc(sizeof(float) * m);
  checkError((h_Ximag != NULL), shrTRUE, "Could not allocate memory");


  for(unsigned i =0; i< m ; i++)
    {
      h_Xreal[i]=0;
      h_Ximag[i]=0;
    }
  h_Xreal[0]=1;
  h_Ximag[0]=1;

  //Precomputation. 

  const float TWOPI = 2*3.14159265358979323846;
  const float theta =  TWOPI / (2 * n);

  for(int l = 0; l < n; l++)
    {
      float c = cos( -1 *theta * l *l);
      float s = sin( -1 *theta * l*l);

      //Toeplitz matrix
      h_Hreal[l] = c;
      h_Himag[l] = s;

      //Y_l Since W_n^-l*l/2
      h_Yreal[l] = h_Xreal[l] * c + h_Ximag[l] * s;
      h_Yimag[l] = h_Ximag[l] *c -  h_Xreal[l] * s;
    }

  for(int i=n; i< m -n +1 ; i++)
    {
      h_Hreal[i] = 0; 
      h_Himag[i] = 0;
      h_Yreal[i] = 0;
      h_Yimag[i] = 0;
    }

  for(int i = m -n +2 ; i < m ; i++)
    {
      h_Hreal[i] = h_Hreal[m-i]; 
      h_Himag[i] = h_Himag[m-i];
      h_Yreal[i] = 0;
      h_Yimag[i] = 0;
    }
}


unsigned 
initExecutionBluesteins(const unsigned size, const unsigned m)
{
  allocateHostMemoryBluesteins(size, m);

      if (deviceCount) {
        printf("Initializing device(s).." );
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
allocateDeviceMemoryBS(const unsigned device, const unsigned size, 
                                      const unsigned copyOffset)
{
    d_Hreal[device] = createDeviceBuffer(
                        CL_MEM_READ_ONLY,
                        sizeof(float) * size,
                        h_Hreal + copyOffset);
    copyToDevice(device, d_Hreal[device],  h_Hreal + copyOffset, size);

    d_Himag[device] = createDeviceBuffer(
                        CL_MEM_READ_ONLY,
                        sizeof(float) * size,
                        h_Himag + copyOffset);
    copyToDevice(device, d_Himag[device],  h_Himag + copyOffset, size);

    d_Yreal[device] = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Yreal + copyOffset);
    copyToDevice(device, d_Yreal[device],  h_Yreal + copyOffset, size);

    d_Yimag[device] = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Yimag + copyOffset);
    copyToDevice(device, d_Yimag[device],  h_Yimag + copyOffset, size);

    d_Zreal[device] = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Zreal + copyOffset);
    copyToDevice(device, d_Zreal[device],  h_Zreal + copyOffset, size);

    d_Zimag[device] = createDeviceBuffer(CL_MEM_WRITE_ONLY,
                                              sizeof(float) * size,
                                              h_Zimag + copyOffset);
    copyToDevice(device, d_Zimag[device],  h_Zimag + copyOffset, size);

}


bool
runBluesteinsFFT(const char * const argv[], const unsigned n,
		 const unsigned size)
{
  //First we need to determine M.
  unsigned M = 1;
  for(M=1; M < 2*n -2; M=M*2);

  if (!initExecutionBluesteins(size, M)) {
         return false;
    }
  bluesteinsFFTGpu(argv, M,n, size);
    return true;
}

void
bluesteinsFFTGpu(const char* const argv[],const unsigned n, 
		 const unsigned orign,const unsigned size)
{
  const unsigned powM = (unsigned) log2(n);
  printf("Compiling Bluesteins Program..\n");

  compileProgram(argv, "fft.h", "kernels/bluesteins.cl");

    printf("Creating Kernel\n");
    for (unsigned i = 0; i < deviceCount; ++i) {
        createKernel(i, "bluesteins");
    }

    const unsigned sizePerGPU = size / deviceCount;
    for (unsigned i = 0; i < deviceCount; ++i) {
        workSize[i] = (i != (deviceCount - 1)) ? sizePerGPU 
                                               : (size - workOffset[i]);       
        
        allocateDeviceMemoryBS(i , workSize[i], workOffset[i]);
        
        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void*) &d_Hreal[i]);
        clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void*) &d_Himag[i]);
	clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void*) &d_Yreal[i]);
        clSetKernelArg(kernel[i], 3, sizeof(cl_mem), (void*) &d_Yimag[i]);
	clSetKernelArg(kernel[i], 4, sizeof(cl_mem), (void*) &d_Zreal[i]);
        clSetKernelArg(kernel[i], 5, sizeof(cl_mem), (void*) &d_Zimag[i]);
	clSetKernelArg(kernel[i], 6, sizeof(unsigned), &n);
	clSetKernelArg(kernel[i], 7, sizeof(unsigned), &orign);
	clSetKernelArg(kernel[i], 8, sizeof(unsigned), &powM);
	clSetKernelArg(kernel[i], 9, sizeof(unsigned), &blockSize);
       

        if ((i + 1) < deviceCount) {
            workOffset[i + 1] = workOffset[i] + workSize[i];
        } 

    }

    size_t localWorkSize[] = {blockSize};
    for (unsigned i = 0; i < deviceCount; ++i) {
        size_t globalWorkSize[] = {shrRoundUp(blockSize, workSize[i])}; 
        // kernel non blocking execution 
        runKernel(i, localWorkSize, globalWorkSize);
    }

    h_Rreal = h_Hreal;
    h_Rimag = h_Himag;
    
    for (unsigned i = 0; i < deviceCount; ++i) {
        copyFromDevice(i, d_Hreal[i], h_Rreal + workOffset[i],
                                                workSize[i]); 
        copyFromDevice(i, d_Himag[i], h_Rimag + workOffset[i],
                                                 workSize[i]);
    }

    // wait for copy event
    const cl_int ciErrNum = clWaitForEvents(deviceCount, gpuDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");
    printGpuTime();
}
