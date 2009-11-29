#include "clutil.h"
#include "fft.h"
#include "kernels.h"


static unsigned workOffset[MAX_GPU_COUNT];
static unsigned workSize[MAX_GPU_COUNT];

bool 
runCooleyTukey(const char* const argv[], const unsigned n, const unsigned size)
{
  if (!initExecution(size, n)) {
         return false;
     }
     unsigned sizeOnGPU = 0;
     unsigned sizeOnCPU = 0; 

     partition(size, sizeOnGPU, sizeOnCPU);

     #pragma omp parallel for 
     for (unsigned i = 0; i < 2; ++i) {
         if ( i == 0) {
             cooleyTukeyGpu(argv, n,  sizeOnGPU);
         } else {
             const unsigned start = sizeOnGPU;
             cooleyTukeyCpu(start, n, sizeOnCPU);
         }      
     }

     return true;
}

void
cooleyTukeyGpu(const char* const argv[], const unsigned n, const unsigned size) 
{
    if (size == 0) return;
    if (deviceCount == 0) return;
    const unsigned powN = (unsigned)log2(n);

    printf("Compiling Cooley Tukey Program..\n");
    compileProgram(argv, "fft.h", "kernels/cooleyTukey.cl");

    for (unsigned i = 0; i < deviceCount; ++i) {
        createKernel(i, "reverse");
    }
    
    const unsigned sizePerGPU = size / deviceCount;
    for (unsigned i = 0; i < deviceCount; ++i) {
        workSize[i] = (i != (deviceCount - 1)) ? sizePerGPU
                                               : (size - workOffset[i]);

        allocateDeviceMemory(i , workSize[i], workOffset[i]);
        clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void*) &d_Freal[i]);
        clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void*) &d_Fimag[i]);
        clSetKernelArg(kernel[i], 2, sizeof(unsigned), &n); 
        clSetKernelArg(kernel[i], 3, sizeof(unsigned), &powN);
        clSetKernelArg(kernel[i], 4, sizeof(unsigned), &blockSize);
	clSetKernelArg(kernel[i], 5, sizeof(unsigned), &size);

        if ((i + 1) < deviceCount) {
            workOffset[i + 1] = workOffset[i] + workSize[i];
        }

    }

    size_t localWorkSize[] = {blockSize};
   
    for (unsigned i = 0; i < deviceCount; ++i) { 
        size_t globalWorkSize[] = {shrRoundUp(blockSize, workSize[i])};
        runKernel(i, localWorkSize, globalWorkSize);
    }

    for (unsigned i = 0; i < deviceCount; ++i) {   
	 copyFromDevice(i, d_Fimag[i], h_Rimag + workOffset[i],
			 workSize[i]);
	 copyFromDevice(i, d_Freal[i], h_Rreal + workOffset[i],
			 workSize[i]);
    }
    
    // wait for copy event
    const cl_int ciErrNum = clWaitForEvents(deviceCount, gpuDone);
    checkError(ciErrNum, CL_SUCCESS, "clWaitForEvents");
    printGpuTime();
}

//   Inplace version of rearrange function
void
cooleyTukeyCpu(const unsigned offset, const unsigned  N, const unsigned size)
{
    if (size == 0) return;
    if (useCpu == 0) return; 
    struct rusage start;
    getrusage(RUSAGE_SELF, &start);
    const unsigned powN = (unsigned)log2(N);
    //TODO:: set the number of threads
    #pragma omp parallel for
    for (unsigned i = 0; i < size; ++i) {
        unsigned int lIndex =  i % N;
        unsigned int lPosition  = 0;
        unsigned int lReverse= 0;
        while(lIndex) {
            lReverse = lReverse << 1;
            lReverse += lIndex %2;
            lIndex = lIndex>>1;
            lPosition++;
        }
        if (lPosition < powN) {
            lReverse = lReverse << (powN-  lPosition);
        }
        lReverse = lReverse + (i / N) * N + offset;
        h_Rreal[lReverse] = h_Freal[i + offset];
        h_Rimag[lReverse] = h_Fimag[i + offset];
    }
    const double twopi =  2 * 3.14159265358979323846;
    for (unsigned i = 0; i < size / N; ++i) {
        for (unsigned p = 0; p < powN ; ++p ) {
            const unsigned powP = (unsigned)pow(2, p);
            #pragma omp parallel for
            for (unsigned k = 0; k < N / 2; ++k) {
                const unsigned indexAdd = i * N + (k /  powP)
                                            * 2 * powP + k % powP + offset;

                const unsigned indexMult = indexAdd + powP;

                const unsigned kk = k % powP;
                const float cs = cos(twopi * kk / ( 2 * powP));
                const float sn = sin(twopi * kk / ( 2 * powP));
                const float addReal = h_Rreal[indexAdd];
                const float addImag = h_Rimag[indexAdd];
                const float tempReal = cs * h_Rreal[indexMult] + 
                                       sn * h_Rimag[indexMult];
                const float tempImag = cs * h_Rimag[indexMult] - 
                                       sn * h_Rreal[indexMult];
                h_Rreal[indexAdd] = addReal + tempReal;
                h_Rimag[indexAdd] = addImag + tempImag;
                h_Rreal[indexMult] =  addReal - tempReal;
                h_Rimag[indexMult] =  addImag - tempImag;
           }
        }
    }
    struct rusage end;
    getrusage(RUSAGE_SELF, &end);
    printCpuTime(start, end);
}
