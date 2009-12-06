/*
   Cooley tukey Implementation of FFT
 */


__kernel void
reverse(__global float* const r_real, __global float* const r_imag,
		const unsigned  n, const unsigned powN,
                const unsigned blockSize,const unsigned size )

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
	lIndex = addr%n;
	if(lReverse > lIndex)
	{
                float lRevTemp;
		const unsigned to = lReverse + (addr / n) * n;
		lRevTemp   = r_real[addr];
		r_real[addr] = r_real[to];
		r_real[to] = lRevTemp;
		lRevTemp   = r_imag[addr];
		r_imag[addr] = r_imag[to];
		r_imag[to]  = lRevTemp;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Now we have to iterate powN times Iteratively

	const unsigned lThread = (addr/(n/2))*n + addr%(n/2);

	if(lThread>size)
		return;
	lIndex =  lThread %n;
	int Iter,nIter;
	for(Iter =0 ,nIter =1;Iter<(powN);Iter ++,nIter*=2)
	{
		const unsigned lIndexAdd  =  (lThread/n)*n + (lIndex/nIter)*2*nIter + lIndex%nIter ;
		const unsigned  lIndexMult =  lIndexAdd +nIter ;
		const unsigned k  = lIndex%nIter;
		const float cs =  cos(TWOPI*k/(2*nIter));
		const float sn =  sin(TWOPI*k/(2*nIter));
		const float add_real =  r_real[lIndexAdd];
		const float add_imag =  r_imag[lIndexAdd] ;
                const float mult_real = r_real[lIndexMult];
                const float mult_imag = r_imag[lIndexMult];
		const float tmp_real = cs*mult_real
                                       + sn * mult_imag;
		const float tmp_imag = cs*mult_imag - sn 
                                       * mult_real; 
		r_real[lIndexAdd] = add_real + tmp_real;
		r_imag[lIndexAdd] = add_imag + tmp_imag;
		r_real[lIndexMult] = add_real - tmp_real;
		r_imag[lIndexMult] = add_imag - tmp_imag;
		barrier(CLK_LOCAL_MEM_FENCE);
	} 
}

