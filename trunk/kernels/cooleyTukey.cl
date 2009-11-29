/*
   Cooley tukey Implementation of FFT
 */


__kernel void
reverse( __global float* f_real, __global float* f_imag,
		__global float* r_real, __global float* r_imag,
		const unsigned  n, const unsigned powN,
                const unsigned blockSize)

{

	const float TWOPI = 2*3.14159265359;
	const size_t bx = get_group_id(0);
	const size_t tx = get_local_id(0);
	const unsigned  addr = bx * blockSize + tx; 
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

	int nIter =1;
	int Iter =0;
	__global float* lTemp =0;
	lIndex =  addr % n;
	for(Iter;Iter<powN;Iter ++)
	{
		// to know which half it is
		int lIndexMult = (lIndex/(2*nIter))*2*nIter +(lIndex%nIter) + nIter + (addr/n)*n  ;
		int lIndexAdd  =(lIndex/(2*nIter))*2*nIter + (lIndex%nIter)  + (addr/n)*n  ; 
		int k  = lIndex%nIter;

		//Multiplying
		float cs =  cos(TWOPI*k/(2*nIter));
		float sn =  sin(TWOPI*k/(2*nIter));
		float tmp_real= cs*r_real[lIndexMult] + sn * r_imag[lIndexMult];
		float tmp_imag = cs*r_imag[lIndexMult] - sn * r_real[lIndexMult];
		int lHalf = (lIndex%(2*nIter))/nIter;
		if(lHalf)
		{
			f_real[addr] = r_real[lIndexAdd] - tmp_real;
			f_imag[addr] = r_imag[lIndexAdd] - tmp_imag;
		}
		else
		{
			f_real[addr] = r_real[lIndexAdd] + tmp_real;
			f_imag[addr] = r_imag[lIndexAdd] + tmp_imag;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		lTemp= f_real;
		f_real= r_real;
		r_real= lTemp;
		lTemp= f_imag;
		f_imag= r_imag;
		r_imag= lTemp;
		nIter*=2;
	}

}

