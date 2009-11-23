#ifndef KERNELS_H
#define KERNELS_H

int
cooleyTukey(const char* const argv[], float* hFreal, float* hFimag,
            float* hRreal, float* hRimag,
            const unsigned n);

int
slowFFT( const char* const argv[], float* hFreal, float* hFimag,
         float* hRreal, float* hRimag,
         const unsigned n, const int is);


#endif
