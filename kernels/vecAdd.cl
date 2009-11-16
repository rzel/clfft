
/* Kernel for the vector add application. */

__kernel void
vecAdd( __global float* A, __global float* B,
	__global float* C)
{

  /* Get block id and thread id. */
  int bx = get_group_id(0);
  int tx = get_local_id(0);

  //Store sum .
  int addr = bx * BLOCK_SIZE + tx;
  if(addr < ARR_SIZE)
    C[addr] = A[addr] + B[addr];
}
 

__kernel void
vecAddTest(void)
{

  /* Get block id and thread id. */
  int bx = get_group_id(0);
  int tx = get_local_id(0);

  //Store sum . 
  int addr = bx * BLOCK_SIZE + tx;
  // C[addr] = 1.0;
  int y = addr * addr;
}
