/*
   Cooley tukey Implementation of FFT
   */


__kernel void
reverse( __global float* f_real, __global float* f_imag,
         __global float* r_real, __global float* r_imag,
         unsigned  n, unsigned powN)

{

    const float TWOPI = 2*3.14159265359;
    const size_t bx = get_group_id(0);
    const size_t tx = get_local_id(0);
    const unsigned  addr = bx * BLOCK_SIZE + tx; 
    //   Swap position
    unsigned int Target = 0;
    //   Process all positions of input signal
    unsigned int lIndex =  addr % n;
    unsigned int lPosition  = 0;
    unsigned int lReverse= 0;
    while(lIndex) {
        lReverse = lReverse << 1;
        lReverse += lIndex %2;
        lIndex = lIndex>>1;
        lPosition++;
    }
    if (lPosition < powN) {
        lReverse =lReverse<<(powN-  lPosition);  
    }
    // The Input vertex will be used as the Buffer vertex
    // We will keep on changing the input and output
    // So tht we dont need to use another Buffer Array
    const unsigned to = lReverse + (addr / n) * n;
    r_real[to] = f_real[addr];
    r_imag[to] = f_imag[addr];

    barrier(CLK_LOCAL_MEM_FENCE);
    // Now we have to iterate powN times Iteratively

    if(addr%2)
    {
	    f_real[addr] = r_real[addr-1] - r_real[addr];
	    f_imag[addr] = r_imag[addr-1] - r_imag[addr];
    }
    else
    {
	    f_real[addr] = r_real[addr+1] + r_real[addr];
	    f_imag[addr] = r_imag[addr+1] + r_imag[addr];

    }
    int nIter =2;
    int Iter =1;
    __global float* lBufreal = f_real;
    __global float* lBufimag = f_imag;
    __global float* lResultreal = r_real;
    __global float* lResultimag = r_imag;
    __global float* lTemp =0;
    lIndex =  addr % n;
    for(Iter;Iter<powN;Iter ++)
    {
	nIter*=2;
	// to know which half it is
	int lIndexMult = lIndex/(2*Iter) +(lIndex%Iter) + Iter + addr%n  ;
	int k  = lIndex%Iter;

	//Multiplying
	float cs =  cos(TWOPI*k/Iter);
	float sn =  sin(TWOPI*k/Iter);
	float tmp_real= cs*lBufreal[lIndexMult] + sn * lBufimag[lIndexMult];
	float tmp_imag = cs*lBufimag[lIndexMult] - sn * lBufreal[lIndexMult];
	int lHalf = lIndex%(2*Iter)/Iter;
	if(lHalf)
	{
		lResultreal[addr] = lBufreal[addr] + tmp_real;
		lResultimag[addr] = lBufimag [addr] + tmp_imag;
	}
	else
	{
		lResultreal[addr] = lBufreal[addr] - tmp_real;
		lResultimag[addr]  = lBufimag [addr] - tmp_imag;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	lTemp= lBufreal ;
	lBufreal = lResultreal;
	lResultreal = lTemp;
	lTemp= lBufimag;
	lBufimag = lResultimag ;
	lResultimag = lTemp;
    }
}

