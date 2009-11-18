
/*
  Slow fft implementation. 
 */

__kernel void
slowfft( __global float* f_real, __global float* f_imag,
	 __global float* r_real, __global float* r_imag,
	int n, int is)
{
  const float PI = 3.14159265359;
  const float ph = is*2.0*PI/n;
  float real = 0.0, imag = 0.0;

  int bx = get_group_id(0);
  int tx = get_local_id(0);

  int addr = bx * BLOCK_SIZE + tx;
  int start = (addr / n)* n;
  int end = (addr / n + 1) * n; 

  float rx, ix,val;

  for(int k=start; k < end; k++)
    {
      rx = f_real[k]; 
      ix = f_imag[k];

      val = ph * (k-start) * (tx % n);
       /* cos(ph*k*w) --> where k from 1 to n and w from 1 to n. */
      real+= rx* cos(val) - ix * sin(val);
      imag+= rx* sin(val) + ix * cos(val);
     
    }

  r_real[addr]= real;
  r_imag[addr]= imag;

}
