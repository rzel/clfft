/* 
   Stockham's implementation of fft.
 */


/* We will have n/2 amount of threads. */

__kernel void
stockham(  __global float * r_real, __global float * r_imag,
	  const unsigned n, const unsigned powN, const unsigned blockSize)
{

  const size_t bx = get_group_id(0);
  const size_t tx = get_local_id(0);
  const unsigned  tid = (bx * blockSize + tx)%(n/2); /*Since n/2 threads are reqd to compute for n. */ 
  const float TWOPI = 1 * 2*3.14159265359;
  const unsigned base = (((bx * blockSize + tx)/(n/2)) * (n/2))*2 ; /*Since n/2 threads compute on n elems at a time.*/


  int l = n/2; 
  int m = 1;
  int j = tid;
  int k=0;

  float theta;

  float c0_real, c0_imag, c1_real, c1_imag;
  float real_diff, imag_diff;

  for(int i =0 ; i < powN; i++)
    {
      k = tid - j*m;
      
      //do main computation here.
      
      theta = TWOPI * j / (2.0 * l);
      
      c0_real = r_real[base + k + j*m];
      c0_imag = r_imag[base + k + j*m];

      c1_real = r_real[base + k + j*m + l*m];
      c1_imag = r_imag[base + k + j*m + l*m];

      //y[k + 2*j*m] = c0 + c1
      
      r_real[base + k + 2*j*m] = c0_real + c1_real;
      r_imag[base + k + 2*j*m] = c0_imag + c1_imag;
      
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;
      
      r_real[base + k + 2*j*m + m] = cos(theta)*(real_diff) - sin(theta)*(imag_diff);
      r_imag[base + k + 2*j*m + m] = cos(theta)*(imag_diff) + sin(theta)*(real_diff);  

      //change values of m,l,j
      j = j/2;
      l = l/2;
      m = m*2;

      barrier(CLK_LOCAL_MEM_FENCE);
    }


}
