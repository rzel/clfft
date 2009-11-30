
/*
  Slow fft implementation. 
 */

__kernel void
slowfft( __global float* f_real, __global float* f_imag,
	 __global float* r_real, __global float* r_imag,
	 const unsigned n, const unsigned blockSize)
{
    const float PI = 3.14159265359;
    const float ph = -1 * 2.0 * PI/n;

    const size_t bx = get_group_id(0);
    const size_t tx = get_local_id(0);

    const int addr = bx * blockSize + tx;
    const int start = (addr / n)* n;
    const int end = (addr / n + 1) * n;


    float real = 0.0, imag = 0.0;
    for (int k = start; k < end; k++) {
        const float rx = f_real[k];
        const float ix = f_imag[k];

        const float val = ph * (k-start) * (addr % n);
	const float cs  = cos(val);
	const float sn  = sin(val);
        /* cos(ph*k*w) --> where k from 1 to n and w from 1 to n. */
        real+= rx* cs - ix * sn;
        imag+= rx* sn + ix * cs;
    }

    r_real[addr]= real;
    r_imag[addr]= imag;
}
