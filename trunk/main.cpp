
#include "clutil.h"

#include "vecAdd.h"

int main(int argc, char * argv[])
{
  cl_uint ciDeviceCount;
  cl_device_id device;
  cl_program  cpProgram;
  cl_kernel kernobj;
  cl_mem d_A, d_B, d_C;
  float *h_A, *h_B, *h_C;
  size_t localWorkSize[] = {BLOCK_SIZE};
  size_t globalWorkSize[]= {shrRoundUp(BLOCK_SIZE, ARR_SIZE + BLOCK_SIZE -1)};


  printf("Initializing CL Context..\n");
  if(init_cl_context(CL_DEVICE_TYPE_GPU) !=CL_SUCCESS)
    {
      printf("Error ! Aborting..\n");
      exit(1);
    }

  printf("Getting Device Count..\n");
  getDeviceCount(&ciDeviceCount);

  if(!ciDeviceCount)
    {
      printf("No opencl specific devices!\n");
      return -1;
    }

  printf("Creating Command Queue...\n");
  createCommandQueue(&device, 1);

  printf("Compiling Program..\n");
  if(compile_program(argv,"vecAdd.h", "kernels/vecAdd.cl", &cpProgram) != CL_SUCCESS)
    {
      printf("Compilation failed.\n");
      return -1;
    }

  if(createKernel(cpProgram,"vecAdd", &kernobj) != CL_SUCCESS)
    {
      printf("Kernel Creation failure.\n");
      return -1;
    }
  
    // Allocate host memory. 

  h_A = (float *) malloc(sizeof(float) * ARR_SIZE);
  if(h_A == NULL)
    {
      printf("Error Could not allocate memory!\n");
      return -1;
    }
  h_B = (float *) malloc(sizeof(float) * ARR_SIZE);
  if(h_B == NULL)
    {
      printf("Error could not allocate memory!\n");
      return -1;
    }
  h_C = (float *) malloc(sizeof(float) * ARR_SIZE);
  if(h_C == NULL)
    {
     printf("Error could not allocate memory!\n");
     return -1;
    }

  printf("Initializing Arrays.. \n");
  for(int i= 0 ; i< ARR_SIZE; i++)
    {
      h_A[i] = 0.0;
      h_B[i] = 1.0;
      h_C[i]= -1.0;
    }


  d_A = createDeviceBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			   sizeof(float) * ARR_SIZE,
			   h_A,
			   true);

  d_B = createDeviceBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			   sizeof(float) * ARR_SIZE,
			   h_B,
			   true);
  
  d_C = createDeviceBuffer(CL_MEM_WRITE_ONLY ,
			   sizeof(float) * ARR_SIZE,
			   h_C,
			   true);

  
  clSetKernelArg(kernobj, 0, sizeof(cl_mem), (void*) &d_A);
  clSetKernelArg(kernobj, 1, sizeof(cl_mem), (void*) &d_B);
  clSetKernelArg(kernobj, 2, sizeof(cl_mem), (void*) &d_C);


  runKernel(kernobj, 1,  		/* Work Dimension. */
	    localWorkSize,
	    globalWorkSize);

  copyfromDevice(d_C, sizeof(float) * ARR_SIZE, h_C, true);

  for(int i=0 ;i <ARR_SIZE; i++ )
    {
      if(h_C[i] != ((float)(1.0)) )
	{
	  printf("Test Failed -- i=%d C[i]=%f\n",i, h_C[i]);
	  for(int j=i - 10 ; j < (i + 100) && (j < ARR_SIZE); j++)
	    {
	      printf("%f ",h_C[j]);
	    }
	  printf("\n");
	  return -1;
	}
    }

  printf("Test Passed !!\n");
  
}