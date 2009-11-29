#ifndef KERNELS_H
#define KERNELS_H

// Driver functions
int
runSlowFFT(const char* const argv[], const unsigned n,
                                             const unsigned size);



// FOR GPU
int
cooleyTukey(const char* const argv[], const unsigned n, const unsigned size);

int
slowFFTGpu(const char* const argv[], const unsigned n,
                                             const unsigned size);

int
stockhamFFT(const char* const argv[], const unsigned n,
                                         const unsigned size);

// FOR CPU
void
slowFFTCpu(const unsigned start, const unsigned N, const unsigned size);



#endif
