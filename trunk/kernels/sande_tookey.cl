/* 
   The Sande-Tookey FFT Algorithm. 
 */

__kernel void
sande_tookey( __global float * r_real, __global float * r_imag,
	  const unsigned n, const unsigned powN, const unsigned blockSize)
{

  const size_t bx = get_group_id(0);
  const size_t tx = get_local_id(0);
  const unsigned  tid = (bx * blockSize + tx)%(n/2); /*Since n/2 threads are reqd to compute for n. */ 
  const float TWOPI = -1 * 2*3.14159265359;
  const unsigned base = (((bx * blockSize + tx)/(n/2)) * (n/2))*2 ; /*Since n/2 threads compute on n elems at a time.*/

  int r,j,mh;
  float theta;

  float u_real, u_imag, v_real, v_imag;
  float real_diff, imag_diff;

  /* i 1 -->  1 to lg2(N/2)
     r=((tid%i) * (N)/i); j=(tid)/i) , mh = (N/2)/i*/
  for(int i=1; i < (powN ); i++)
    {

      r = ((tid%i) * (n/i));
      j = tid/i;
      mh = (n/2) / i;

      theta = TWOPI * j / (2 * mh);

      u_real = r_real[base + r + j];
      u_imag = r_imag[base + r + j];

      v_real = r_real[base + r + j + mh];
      v_imag = r_imag[base + r + j + mh];

      r_real[base + r + j] = u_real + v_real;
      r_imag[base + r + j] = u_imag + v_imag;

      real_diff = u_real - v_real;
      imag_diff = u_imag - v_imag;

      r_real[base + r + j + mh] = cos(theta)*(real_diff) - sin(theta)*(imag_diff);
      r_imag[base + r + j + mh]=  cos(theta)*(imag_diff) + sin(theta)*(real_diff);  
      

      barrier(CLK_LOCAL_MEM_FENCE);
    }
            
}
