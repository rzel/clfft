/*
   Cooley tukey Implementation of FFT
   */


__kernel void
reverse( __global float* f_real, __global float* f_imag,
         __global float* r_real, __global float* r_imag,
         unsigned  n, unsigned powN)

{
    const size_t bx = get_group_id(0);
    const size_t tx = get_local_id(0);
    const unsigned  addr = bx * BLOCK_SIZE + tx; 
    //   Swap position
    unsigned int Target = 0;
    //   Process all positions of input signal
    unsigned int lIndex =  addr % n;
    unsigned int lPosition  = 0;
    unsigned int lReverse= 0;
    unsigned int lTemp = 0;
    while(lIndex) {
        lReverse = lReverse << 1;
        lReverse += lIndex %2;
        lIndex = lIndex>>1;
        lPosition++;
    }
    if (lPosition < powN) {
        lReverse =lReverse<<(powN-  lPosition);  
    }

    if (lReverse <  (addr % n)) {
        const unsigned to = lReverse + (addr / n) * n;
        float temp = f_real[addr];
        f_real[addr] = f_real[to];
        f_real[to] = temp;

        temp = f_imag[addr];
        f_imag[addr] = f_imag[to];
        f_imag[to] = temp;
        
    }
    
}

