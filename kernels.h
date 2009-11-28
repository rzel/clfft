#ifndef KERNELS_H
#define KERNELS_H

int
cooleyTukey(const char* const argv[], const unsigned n, const unsigned size);

int
slowFFT(const char* const argv[], const unsigned n, const int is, 
                                             const unsigned size);

int
stockhamFFT(const char* const argv[], const unsigned n, const int is, 
                                             const unsigned size);

#endif
