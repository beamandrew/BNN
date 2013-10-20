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
__global__ void add_ARD_grad(float *gradW, float* weights, float *sig, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        gradW[index] -= weights[index]/(sig[row]*sig[row]);          
        }                        
}

//Add prior contribution to gradient
__global__ void add_normal_unit_grad(float *gradW, float* weights, float *sig, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        gradW[index] -= weights[index]/(sig[col]*sig[col]);          
        }                        
}

//Add prior contribution to gradient
__global__ void add_bias_grad(float *gradB, float* biases, float *sig, int M, int N)
{
        int index = blockIdx.x*blockDim.x + threadIdx.x;
                
        if(index < N) {
	        gradB[index] -= biases[index]/(sig[0]*sig[0]);          
        }                        
}

//Scale momentum variables using current sd for ARD prior
__global__ void scale_momentum_ARD(float* p, float *sig, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        p[index] = p[index]*(sig[row]*sig[row]);          
        }                        
}     

//Scale momentum variables using current sd for normal unit prior
__global__ void scale_momentum_normal_unit(float* p, float *sig, int M, int N)
{
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int index = row*N + col;
                
        if(row < M && col < N) {
	        p[index] = p[index]*(sig[col]*sig[col]);          
        }                        
}
