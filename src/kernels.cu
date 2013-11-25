//Softmax function
__global__ void softmax(float *output, int M, int N)
{
	#include <math.h>
	//row is index into grad_h
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        //int col = blockIdx.x*blockDim.x + threadIdx.x;
        //int index = row*N + col;
        float sum = 0;     
        if(row < M) {// && col < N) {
		for(int i=0;i<N;i++){
			sum += exp(output[row*N + i]);
		}
        	for(int i=0;i<N;i++){
			output[row*N + i] = exp(output[row*N + i])/sum;
		}	           
        }                 
                  
}

//Add bias to dot-product using GPU
__global__ void add_bias(float *output, float *bias, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        output[index] += bias[col];          
        }                        
}

//Add prior contribution to gradient
__global__ void add_ARD_grad(float *gradW, float* weights, float *sig2, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        gradW[index] -= weights[index]/(sig2[row]);          
        }                        
}

//Add prior contribution to gradient with sigma integrated out
__global__ void add_ARD_grad_ALT(float *gradW, float* weights, float shape, float scale, float Nw, float wSS,
                                    int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        gradW[index] -= weights[index]*(shape*Nw + 2*scale)/(scale*wSS);          
        }                        
}


//Add prior contribution to gradient
__global__ void add_gaussian_layer_grad(float *gradW, float* weights, float *sig2, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        gradW[index] -= weights[index]/(sig2[0]);          
        }                        
}

//Add prior contribution to gradient
__global__ void add_gaussian_unit_grad(float *gradW, float* weights, float *sig2, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        gradW[index] -= weights[index]/(sig2[col]);          
        }                        
}

//Add prior contribution to gradient
__global__ void add_bias_grad(float *gradB, float* biases, float *sig2, int M, int N)
{
        int index = blockIdx.x*blockDim.x + threadIdx.x;
                
        if(index < N) {
	        gradB[index] -= biases[index]/(sig2[0]);          
        }                        
}

//Scale momentum variables using current sd for ARD prior
__global__ void scale_momentum_ARD(float* p, float *sig2, int M, int N)
{
        #include <math.h>
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
             float val = p[index]*sqrt(sig2[row]);
	        p[index] = val > 10 ? 10 : val;          
        }                        
}     

//Scale momentum variables using current sd for ARD prior
__global__ void scale_stepsize_ARD(float* epsW, float *sig2, int M, int N)
{
        #include <math.h>
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
             float val = epsW[index]*sqrt(sig2[row]);
	        epsW[index] =  val > 10 ? 10 : val;          
        }                        
}     

//Scale momentum variables using current sd for ARD prior
__global__ void scale_momentum_Gaussian_Layer(float* p, float *sig2, int M, int N)
{
        #include <math.h>
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
             float val = p[index]*sqrt(sig2[0]);
	        p[index] = val > 10 ? 10 : val;          
        }                        
}     

//Scale momentum variables using current sd for ARD prior
__global__ void scale_stepsize_Gaussian_Layer(float* epsW, float *sig2, int M, int N)
{
        #include <math.h>
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
              float val = epsW[index]*sqrt(sig2[0]);
	        epsW[index] = val > 10 ? 10 : val;                
        }                        
}    


//Scale momentum variables using current sd for normal unit prior
__global__ void scale_momentum_normal_unit(float* p, float *sig2, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
              float val = p[index]*(sig2[col]);
	        p[index] = val > 10 ? 10 : val;                
        }                        
}

__global__ void rect_grad(float* back_prop_signal, float *h, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
              if(h[index] == 0.0) {
                back_prop_signal[index] = 0;
              }                
        }                        
}

