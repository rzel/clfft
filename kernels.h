#ifndef KERNELS_H
#define KERNELS_H

// Driver functions
bool
runSlowFFT(const char* const argv[], const unsigned n,
                                             const unsigned size);

bool
runCooleyTukey(const char* const argv[], const unsigned n,
                                         const unsigned size);


bool
runStockhamFFT(const char* const argv[], const unsigned n,

                                         const unsigned size);



// FOR GPU
void
cooleyTukeyGpu(const char* const argv[], const unsigned n, const unsigned size);

void
slowFFTGpu(const char* const argv[], const unsigned n,
                                             const unsigned size);

void
stockhamFFTGpu(const char* const argv[], const unsigned n,
                                         const unsigned size);

bool
runSande_tookeyFFT(const char* const argv[], const unsigned n,
                                         const unsigned size);

int
sande_tookeyFFTGpu(const char* const argv[], const unsigned n,
		   const unsigned size);

bool
runBluesteinsFFT(const char * const argv[], const unsigned n,
		 const unsigned size);

void
bluesteinsFFTGpu(const char* const argv[],const unsigned n, 
		 const unsigned orign,const unsigned size);

// FOR CPU
void
slowFFTCpu(const unsigned start, const unsigned N, const unsigned size);

void
cooleyTukeyCpu(const unsigned offset, const unsigned int size, const unsigned N);


#endif
