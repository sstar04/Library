 
 
 // CUDA kernel to add elements of two arrays
    __global__
    void init(unsigned int n, bool *x)
    {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
     
      if(index < n){
        x[index] = 0;
        }
    }
     
