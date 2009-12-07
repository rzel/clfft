/* 
   Bluesteins algorithm of FFT for any n . 
   Works for any N though not optimal for any given n.
   Results will be stored in H.
*/

__kernel void
bluesteins( __global float * h_real, __global float * h_imag,
	    __global float * y_real, __global float * y_imag,
	    __global float * z_real, __global float * z_imag,
	    const unsigned M, const unsigned n, const unsigned powM,
	    const unsigned blockSize)

{

  const size_t bx = get_group_id(0);
  const size_t tx = get_local_id(0);
  const unsigned  tid = (bx * blockSize + tx)%(M/2); /*Since n/2 threads are reqd to compute for n. */ 
  const float TWOPI = 2*3.14159265359;
  const unsigned base = (((bx * blockSize + tx)/(M/2)) * (M/2))*2 ; /*Since n/2 threads compute on n elems at a time.*/

  unsigned l = M/2;
  unsigned m = 1;
  unsigned j = tid;
  unsigned k = 0;

  const float theta_const = TWOPI / (2.0 * M);
  float theta;
  float c0_real, c0_imag, c1_real, c1_imag;
  float real_diff, imag_diff;

  float c,s;

  unsigned base1, base2, base3, base4;
  float tid2 = 2 * tid;

  for(int i =0; i< powM; ++i)
    {

      k = tid - j*m;
      theta = theta_const * j;
      base1 = base + k + j*m;
      base2 = base1 + l*m;
      base3 = base1 + j*m;
      base4 = base3 + m;
      // Compute FFT of Toeplitz Matrix i.e H

      c0_real = h_real[base1];
      c0_imag = h_imag[base1];

      c1_real = h_real[base2];
      c1_imag = h_imag[base2];

      h_real[base3] = c0_real + c1_real;
      h_imag[base3] = c0_imag + c1_imag;
      
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;

      c = cos(theta);
      s = sin(theta);
      
      h_real[base4] = c*(real_diff) - s*(imag_diff);
      h_imag[base4] = c*(imag_diff) + s*(real_diff);  

      // Compute FFT of Y_l 

      c0_real = y_real[base1];
      c0_imag = y_imag[base1];

      c1_real = y_real[base2];
      c1_imag = y_imag[base2];

      y_real[base3] = c0_real + c1_real;
      y_imag[base3] = c0_imag + c1_imag;
      
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;

      y_real[base4] = c*(real_diff) - s*(imag_diff);
      y_imag[base4] = c*(imag_diff) + s*(real_diff);  

      j = j/2;
      l = l/2;
      m = m*2;

      barrier(CLK_LOCAL_MEM_FENCE);
    }

  //Compute Z_r 
  z_real[tid]= h_real[tid]*y_real[tid] - h_imag[tid]*y_imag[tid];
  z_imag[tid]= h_real[tid]*y_imag[tid] + h_imag[tid]*y_real[tid];

  z_real[tid2]= h_real[tid2]*y_real[tid2] - h_imag[tid2]*y_imag[tid2];
  z_imag[tid2]= h_real[tid2]*y_imag[tid2] + h_imag[tid2]*y_real[tid2];
  
  barrier(CLK_LOCAL_MEM_FENCE);

   l = M/2; 
   m = 1;
   j = tid;
   k=0;
  
   for(int i =0; i< powM; ++i)
    {
      k = tid - j*m;
      theta = theta_const * j;
      base1 = base + k + j*m;
      base2 = base1 + l*m;
      base3 = base1 + j*m;
      base4 = base3 + m;
      // Compute FFT of Toeplitz Matrix i.e H

      c0_real = z_real[base1];
      c0_imag = z_imag[base1];

      c1_real = z_real[base2];
      c1_imag = z_imag[base2];

      z_real[base3] = c0_real + c1_real;
      z_imag[base3] = c0_imag + c1_imag;
      
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;

      c = cos(theta);
      s = -1 * sin(theta); //Since Inverse.
      
      z_real[base4] = c*(real_diff) - s*(imag_diff);
      z_imag[base4] = c*(imag_diff) + s*(real_diff);  
      
      barrier(CLK_LOCAL_MEM_FENCE);
    }
   
   // Now we get X_r  which is stored in H_r
   c = cos(tid * tid / 2 );
   s = sin(tid * tid / 2 );

   h_real[tid] = (c*z_real[tid] - s*z_imag[tid])/M;
   h_imag[tid] = (c*z_imag[tid] + s*z_read[tid])/M;  

   c = cos(tid * tid * 2 );
   s = sin(tid * tid * 2 );

   h_real[tid2] = (c*z_real[tid2] - s*z_imag[tid2])/M;
   h_imag[tid2] = (c*z_imag[tid2] + s*z_read[tid2])/M;  

}
