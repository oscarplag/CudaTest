#include "2DKerns.cuh"
//#include <math>
#include "math_constants.h"
#include "windows.h"

#define KERNEL_RADIUS 4
#define TILE_W 16

__global__ void kernel(unsigned short* input_image, unsigned short* output_image, int width, int height, float* d_Kernel, int kernSize)
{
		int index_x = blockIdx.x*blockDim.x + threadIdx.x;
		int index_y = blockIdx.y*blockDim.y + threadIdx.y;

		//map the two 2D indices to a single linear 1D index
		int grid_width = gridDim.x*blockDim.x;
		int index = index_y*grid_width + index_x;

		unsigned short value;
		long sum = 0;
		
		int kernRad = kernSize/2;
		for (int i = -kernRad; i <= kernRad; i++)
		{
			for (int j = -kernRad; j <= kernRad; j++)	// col wise
			{
				// check row first
				if (blockIdx.x == 0 && (threadIdx.x + i) < 0)	// left apron
					value = 0;
				else if ( blockIdx.x == (gridDim.x - 1) && (threadIdx.x + i) > blockDim.x-1 )	// right apron
					value = 0;
				else 
				{ 
					// check col next
					if (blockIdx.y == 0 && (threadIdx.y + j) < 0)	// top apron
						value = 0;
					else if ( blockIdx.y == (gridDim.y - 1) &&
						(threadIdx.y + j) > blockDim.y-1 )	// bottom apron
						value = 0;
					else	// safe case
						value = input_image[index + i + j * width];
				} 
				sum += value * d_Kernel[kernRad + i] * d_Kernel[kernRad + j];
			}
		}
		output_image[index] = sum;
	
}

__global__ void kernelShared(unsigned short* input_image, unsigned short* output_image, int width, int height, float* d_Kernel)
{
	int index_x = blockIdx.x*blockDim.x + threadIdx.x;
	int index_y = blockIdx.y*blockDim.y + threadIdx.y;
	//map the two 2D indices to a single linear 1D index
	int grid_width = gridDim.x*blockDim.x;
	int index = index_y*grid_width + index_x;

		
	__shared__ float cache[TILE_W+(2*KERNEL_RADIUS)][TILE_W+(2*KERNEL_RADIUS)];

	int x = index_x-KERNEL_RADIUS;
	int y = index_y-KERNEL_RADIUS;
	if( x < 0 || y < 0)
		cache[threadIdx.x][threadIdx.y] = 0;
	else
		cache[threadIdx.x][threadIdx.y] = input_image[index-KERNEL_RADIUS-width*KERNEL_RADIUS];

	x = index_x+KERNEL_RADIUS;
	y = index_y-KERNEL_RADIUS;
	if( x >= width-1 || y < 0)
		cache[threadIdx.x + blockDim.x][threadIdx.y] = 0;
	else
		cache[threadIdx.x + blockDim.x][threadIdx.y] = input_image[index+KERNEL_RADIUS-width*KERNEL_RADIUS];
		


	x = index_x-KERNEL_RADIUS;
	y = index_y+KERNEL_RADIUS;

	if( x < 0 || y >= height)
		cache[threadIdx.x][threadIdx.y + blockDim.y] = 0;
	else
		cache[threadIdx.x][threadIdx.y + blockDim.y] = input_image[index-KERNEL_RADIUS+width*KERNEL_RADIUS];
		
	x = index_x+KERNEL_RADIUS;

	y = index_y+KERNEL_RADIUS;
	if( x >= width || y >= height)
		cache[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = 0;
	else
		cache[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = input_image[index+KERNEL_RADIUS+width*KERNEL_RADIUS];
		
	__syncthreads();
	//output_image[index] = input_image[index];

	float sum = 0.0;

	x = KERNEL_RADIUS + threadIdx.x;
	y = KERNEL_RADIUS + threadIdx.y;
	for(int i = -KERNEL_RADIUS;i<=KERNEL_RADIUS;++i)
	{
		sum += cache[x+i][y]*d_Kernel[KERNEL_RADIUS+i];
	}
	for(int j = -KERNEL_RADIUS;j<=KERNEL_RADIUS;++j)
	{
		sum += cache[x][y+j]*d_Kernel[KERNEL_RADIUS+j];
	}
	/*for(int i = -KERNEL_RADIUS; i<=KERNEL_RADIUS; ++i)
	{
		for(int j = -KERNEL_RADIUS; i<=KERNEL_RADIUS; ++j)
		{
			//sum += cache[x+i][y+j]*d_Kernel[KERNEL_RADIUS+i]*d_Kernel[KERNEL_RADIUS+j];
			sum += 4500*d_Kernel[KERNEL_RADIUS+j];
		}
	}*/
	sum /=2;
	output_image[index] = unsigned short(sum);	
}

__global__ void kernelSharedCustom(unsigned short* input_image, unsigned short* output_image, int width, int height, float* d_Kernel)
{
	int x;
	int y;
	int index_x = blockIdx.x*blockDim.x + threadIdx.x;
	int index_y = blockIdx.y*blockDim.y + threadIdx.y;
	//map the two 2D indices to a single linear 1D index
	int grid_width = gridDim.x*blockDim.x;
	int index = index_y*grid_width + index_x;

	__shared__ float cache[TILE_W+(2*KERNEL_RADIUS)][TILE_W];
	__shared__ float cache2[TILE_W][TILE_W+(2*KERNEL_RADIUS)];

	
	cache[threadIdx.x+KERNEL_RADIUS][threadIdx.y] = input_image[index];
	cache2[threadIdx.x][threadIdx.y+KERNEL_RADIUS] = input_image[index];

	
	x = index_x-KERNEL_RADIUS;		
	if(threadIdx.x<KERNEL_RADIUS)
	{
		if(x<0)
			cache[threadIdx.x][threadIdx.y]=0;
		else
			cache[threadIdx.x][threadIdx.y] = input_image[index-KERNEL_RADIUS];
	}		
	x = index_x+KERNEL_RADIUS;
	if(threadIdx.x>=(blockDim.x-KERNEL_RADIUS))
	{
		if(x>=width)
			cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y]=0;
		else
			cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y]= input_image[index+KERNEL_RADIUS];
	}
	
	y = index_y-KERNEL_RADIUS;
	if(threadIdx.y<KERNEL_RADIUS)
	{
		if(y<0)
			cache2[threadIdx.x][threadIdx.y]=0;
		else
			cache2[threadIdx.x][threadIdx.y] = input_image[index-width*KERNEL_RADIUS];
	}
	y=index_y+KERNEL_RADIUS;
	if(threadIdx.y>=(blockDim.y-KERNEL_RADIUS))
	{
		if(y>=height)
			cache2[threadIdx.x][threadIdx.y+2*KERNEL_RADIUS] = 0;
		else
			cache2[threadIdx.x][threadIdx.y+2*KERNEL_RADIUS] = input_image[index+width*KERNEL_RADIUS];
	}

	__syncthreads();
	
	//output_image[index] = input_image[index];
	
	float sum = 0.0;

	x = KERNEL_RADIUS + threadIdx.x;
	y = threadIdx.y;//KERNEL_RADIUS + threadIdx.y;
	for(int i = -KERNEL_RADIUS;i<=KERNEL_RADIUS;++i)
	{
		sum += cache[x+i][y]*d_Kernel[KERNEL_RADIUS+i];
	}		
	x = threadIdx.x;
	y = threadIdx.y+KERNEL_RADIUS;
	for(int j = -KERNEL_RADIUS;j<=KERNEL_RADIUS;++j)
	{
		sum += cache2[x][y+j]*d_Kernel[KERNEL_RADIUS+j];
	}
	sum /=2;
	output_image[index] = unsigned short(sum);
}

__global__ void kernelSharedCustom2(unsigned short* input_image, unsigned short* output_image, int width, int height, float* d_Kernel)
{
	int x;
	int y;
	int index_x = blockIdx.x*blockDim.x + threadIdx.x;
	int index_y = blockIdx.y*blockDim.y + threadIdx.y;
	//map the two 2D indices to a single linear 1D index
	int grid_width = gridDim.x*blockDim.x;
	int index = index_y*grid_width + index_x;

	__shared__ float cache[TILE_W+(2*KERNEL_RADIUS)][TILE_W+(2*KERNEL_RADIUS)];
	
	cache[threadIdx.x+KERNEL_RADIUS][threadIdx.y+KERNEL_RADIUS] = input_image[index];
	
	if(threadIdx.x<KERNEL_RADIUS)
	{
		x = index_x-KERNEL_RADIUS;
		if (threadIdx.y<KERNEL_RADIUS)
		{
			y = index_y-KERNEL_RADIUS;
			if(x<0 || y<0)
				cache[threadIdx.x][threadIdx.y]=0;
			else
				cache[threadIdx.x][threadIdx.y] = input_image[index-KERNEL_RADIUS - width*KERNEL_RADIUS];
		}
		if (threadIdx.y>=(blockDim.y-KERNEL_RADIUS))
		{
			y = index_y+KERNEL_RADIUS;
			if(x<0 || y>=height)
				cache[threadIdx.x][threadIdx.y+2*KERNEL_RADIUS]=0;
			else
				cache[threadIdx.x][threadIdx.y+2*KERNEL_RADIUS] = input_image[index-KERNEL_RADIUS + width*KERNEL_RADIUS];
		}
		if(x<0)
			cache[threadIdx.x][threadIdx.y+KERNEL_RADIUS]=0;
		else
			cache[threadIdx.x][threadIdx.y+KERNEL_RADIUS] = input_image[index-KERNEL_RADIUS];
	}		
	if(threadIdx.x>=(blockDim.x-KERNEL_RADIUS))
	{
		x = index_x+KERNEL_RADIUS;

		if(threadIdx.y<KERNEL_RADIUS)
		{
			y = index_y-KERNEL_RADIUS;
			if(x>=width || y<0)
				cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y]=0;
			else
				cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y]=input_image[index+KERNEL_RADIUS-width*KERNEL_RADIUS];
		}
		if (threadIdx.y>=height)
		{
			y = index_y+KERNEL_RADIUS;
			if(x>=width || y>=(blockDim.x-KERNEL_RADIUS))
				cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y+2*KERNEL_RADIUS]=0;
			else
				cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y+2*KERNEL_RADIUS] = input_image[index+KERNEL_RADIUS + width*KERNEL_RADIUS];
		}		
		
		if(x>=width)
			cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y+KERNEL_RADIUS]=0;
		else
			cache[threadIdx.x+2*KERNEL_RADIUS][threadIdx.y+KERNEL_RADIUS]= input_image[index+KERNEL_RADIUS];
		
	}
	if(threadIdx.y<KERNEL_RADIUS)
	{
		y = index_y-KERNEL_RADIUS;
		if(y<0)
			cache[threadIdx.x+KERNEL_RADIUS][threadIdx.y]=0;
		else
			cache[threadIdx.x+KERNEL_RADIUS][threadIdx.y] = input_image[index-width*KERNEL_RADIUS];
	}	
	if(threadIdx.y>=(blockDim.y-KERNEL_RADIUS))
	{
		y=index_y+KERNEL_RADIUS;
		if(y>=height)
			cache[threadIdx.x+KERNEL_RADIUS][threadIdx.y+2*KERNEL_RADIUS] = 0;
		else
			cache[threadIdx.x+KERNEL_RADIUS][threadIdx.y+2*KERNEL_RADIUS] = input_image[index+width*KERNEL_RADIUS];
	}

	__syncthreads();
	
	//output_image[index] = input_image[index];
	
	float sum = 0.0;

	x = KERNEL_RADIUS + threadIdx.x;
	y = KERNEL_RADIUS + threadIdx.y;
	for(int i = -KERNEL_RADIUS;i<=KERNEL_RADIUS;++i)
	{
		sum += cache[x+i][y]*d_Kernel[KERNEL_RADIUS+i];
	}		
	for(int j = -KERNEL_RADIUS;j<=KERNEL_RADIUS;++j)
	{
		sum += cache[x][y+j]*d_Kernel[KERNEL_RADIUS+j];
	}
	sum /=2;
	output_image[index] = unsigned short(sum);
}

__global__ void kernelSharedCustomDynamic(unsigned short* input_image, unsigned short* output_image, int width, int height, float* d_Kernel, int kernRadius)
{
	int x;
	int y;
	int index_x = blockIdx.x*blockDim.x + threadIdx.x;
	int index_y = blockIdx.y*blockDim.y + threadIdx.y;
	//map the two 2D indices to a single linear 1D index
	int grid_width = gridDim.x*blockDim.x;
	int index = index_y*grid_width + index_x;
	int cacheWidth = 2*kernRadius+blockDim.x;

	extern __shared__ float cache[];
	cache[threadIdx.x+KERNEL_RADIUS + cacheWidth*(threadIdx.y+KERNEL_RADIUS)] = input_image[index];
	
	
	if(threadIdx.x<KERNEL_RADIUS)
	{
		x = index_x-KERNEL_RADIUS;
		if (threadIdx.y<KERNEL_RADIUS)
		{
			y = index_y-KERNEL_RADIUS;
			if(x<0 || y<0)
				cache[threadIdx.x+cacheWidth*threadIdx.y]=0;
			else
				cache[threadIdx.x+cacheWidth*threadIdx.y] = input_image[index-KERNEL_RADIUS - width*KERNEL_RADIUS];
		}
		if (threadIdx.y>=(blockDim.y-KERNEL_RADIUS))
		{
			y = index_y+KERNEL_RADIUS;
			if(x<0 || y>=height)
				cache[threadIdx.x+cacheWidth*(threadIdx.y+2*KERNEL_RADIUS)]=0;
			else
				cache[threadIdx.x+cacheWidth*(threadIdx.y+2*KERNEL_RADIUS)] = input_image[index-KERNEL_RADIUS + width*KERNEL_RADIUS];
		}
		if(x<0)
			cache[threadIdx.x+cacheWidth*(threadIdx.y+KERNEL_RADIUS)]=0;
		else
			cache[threadIdx.x+cacheWidth*(threadIdx.y+KERNEL_RADIUS)] = input_image[index-KERNEL_RADIUS];
	}		
	if(threadIdx.x>=(blockDim.x-KERNEL_RADIUS))
	{
		x = index_x+KERNEL_RADIUS;

		if(threadIdx.y<KERNEL_RADIUS)
		{
			y = index_y-KERNEL_RADIUS;
			if(x>=width || y<0)
				cache[threadIdx.x+2*KERNEL_RADIUS+cacheWidth*threadIdx.y]=0;
			else
				cache[threadIdx.x+2*KERNEL_RADIUS+cacheWidth*threadIdx.y]=input_image[index+KERNEL_RADIUS-width*KERNEL_RADIUS];
		}
		if (threadIdx.y>=(blockDim.y-KERNEL_RADIUS))
		{
			y = index_y+KERNEL_RADIUS;
			if(x>=width || y>=height)
				cache[threadIdx.x+2*KERNEL_RADIUS+cacheWidth*(threadIdx.y+2*KERNEL_RADIUS)]=0;
			else
				cache[threadIdx.x+2*KERNEL_RADIUS+cacheWidth*(threadIdx.y+2*KERNEL_RADIUS)] = input_image[index+KERNEL_RADIUS + width*KERNEL_RADIUS];
		}		
		
		if(x>=width)
			cache[threadIdx.x+2*KERNEL_RADIUS+cacheWidth*(threadIdx.y+KERNEL_RADIUS)]=0;
		else
			cache[threadIdx.x+2*KERNEL_RADIUS+cacheWidth*(threadIdx.y+KERNEL_RADIUS)]= input_image[index+KERNEL_RADIUS];
		
	}
	if(threadIdx.y<KERNEL_RADIUS)
	{
		y = index_y-KERNEL_RADIUS;
		if(y<0)
			cache[threadIdx.x+KERNEL_RADIUS+cacheWidth*threadIdx.y]=0;
		else
			cache[threadIdx.x+KERNEL_RADIUS+cacheWidth*threadIdx.y] = input_image[index-width*KERNEL_RADIUS];
	}	
	if(threadIdx.y>=(blockDim.y-KERNEL_RADIUS))
	{
		y=index_y+KERNEL_RADIUS;
		if(y>=height)
			cache[threadIdx.x+KERNEL_RADIUS+cacheWidth*(threadIdx.y+2*KERNEL_RADIUS)] = 0;
		else
			cache[threadIdx.x+KERNEL_RADIUS+cacheWidth*(threadIdx.y+2*KERNEL_RADIUS)] = input_image[index+width*KERNEL_RADIUS];
	}

	__syncthreads();
	
	//output_image[index] = input_image[index];
	
	float sum = 0.0;

	x = KERNEL_RADIUS + threadIdx.x;
	y = KERNEL_RADIUS + threadIdx.y;
	for(int i = -KERNEL_RADIUS;i<=KERNEL_RADIUS;++i)
	{
		sum += cache[x+i+cacheWidth*y]*d_Kernel[KERNEL_RADIUS+i];
	}		
	for(int j = -KERNEL_RADIUS;j<=KERNEL_RADIUS;++j)
	{
		sum += cache[x+cacheWidth*(y+j)]*d_Kernel[KERNEL_RADIUS+j];
	}
	sum /=2;
	output_image[index] = unsigned short(sum);
}

int main(void)
{
	LARGE_INTEGER frequency;
	LARGE_INTEGER t1, t2, t3, t4;
	double elapsedTime;

	unsigned short num_elements_x = 1536;
	
	unsigned short num_elements_y = 1536;
	int kernSize = 2*KERNEL_RADIUS+1;
	float sigma = 10.0;


	int num_bytes = num_elements_x*num_elements_y*sizeof(unsigned short);
	int kern_bytes = kernSize*sizeof(float);

	FILE *fin = fopen("input.raw","r");
	if(fin==NULL)
	{
		printf("Could Not Find Dark Image!\n");
		return -1;
	}

	unsigned short* host_array = (unsigned short*)malloc(num_bytes);
	unsigned short* host_array2 = (unsigned short*)malloc(num_bytes);
	unsigned short* host_array3 = (unsigned short*)malloc(num_bytes);
	float* host_kern = (float*)malloc(kern_bytes);
	
	float sum = 0.0;
	for(int i = 0; i<kernSize; i++)
	{
		int x = i-kernSize/2;
		float temp = 1/(sqrt(2*CUDART_PI_F *sigma))*exp((-1*x*x)/(2*sigma*sigma));
		sum += temp;
		host_kern[i] = temp;
		//printf("kernel at %d (x = %d): %f\n", i, x, temp);
	}

	for(int i = 0; i<kernSize; i++)
	{
		host_kern[i] /= sum;
	}

	
	size_t read =fread(host_array,num_bytes,1,fin);
	fclose(fin);
	printf("Image Loaded...\n");

	unsigned short* device_array_in = 0;
	unsigned short* device_array_out = 0;
	float* device_kern = 0;

	//allocate memory in either space
	cudaMalloc((void**)&device_array_in,num_bytes);
	cudaMalloc((void**)&device_array_out,num_bytes);
	cudaMalloc((void**)&device_kern,kern_bytes);

	unsigned short* device_array_in2 = 0;
	unsigned short* device_array_out2 = 0;
	float* device_kern2 = 0;
	cudaMalloc((void**)&device_array_in2,num_bytes);
	cudaMalloc((void**)&device_array_out2,num_bytes);
	cudaMalloc((void**)&device_kern2,kern_bytes);

	cudaMemcpy(device_array_in,host_array,num_bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(device_kern,host_kern,kern_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_in2,host_array,num_bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(device_kern2,host_kern,kern_bytes, cudaMemcpyHostToDevice);
	
	
	//create two dimensional 4x4 thread blocks
	dim3 block_size;
	block_size.x = TILE_W;
	block_size.y = TILE_W;

	//configure a two dimensional grid as well
	dim3 grid_size;
	grid_size.x = num_elements_x/block_size.x;
	grid_size.y = num_elements_y/block_size.y;

	//grid_size & block_size are passed as arguments to the triple chevrons
	/*int sharedMemory = (block_size.x+2*KERNEL_RADIUS)*(block_size.y+2*KERNEL_RADIUS)*sizeof(float);
	kernelSharedCustomDynamic<<<grid_size,block_size,sharedMemory>>>(device_array_in,device_array_out,num_elements_x,num_elements_y,device_kern,KERNEL_RADIUS);
	printf("Cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(host_array2,device_array_out,num_bytes,cudaMemcpyDeviceToHost);
	FILE* fout = fopen("Dynamic-output.raw","wb");
	int written = fwrite(host_array2,sizeof(unsigned short),num_bytes/2,fout);
	fclose(fout);*/

	/*
	kernelSharedCustom2<<<grid_size,block_size>>>(device_array_in,device_array_out,num_elements_x,num_elements_y,device_kern);
	printf("Cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(host_array2,device_array_out,num_bytes,cudaMemcpyDeviceToHost);
	FILE* fout = fopen("Custom-SingleCache-output.raw","wb");
	int written = fwrite(host_array2,sizeof(unsigned short),num_bytes/2,fout);
	fclose(fout);*/

	/*
	kernelSharedCustom<<<grid_size,block_size>>>(device_array_in,device_array_out,num_elements_x,num_elements_y,device_kern);
	printf("Cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(host_array2,device_array_out,num_bytes,cudaMemcpyDeviceToHost);
	FILE* fout = fopen("Custom-output.raw","wb");
	int written = fwrite(host_array2,sizeof(unsigned short),num_bytes/2,fout);
	fclose(fout);*/

	/*
	kernelShared<<<grid_size,block_size>>>(device_array_in,device_array_out,num_elements_x,num_elements_y,device_kern);
	printf("Cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(host_array2,device_array_out,num_bytes,cudaMemcpyDeviceToHost);
	FILE* fout = fopen("Standard-output.raw","wb");
	int written = fwrite(host_array2,sizeof(unsigned short),num_bytes/2,fout);
	fclose(fout);*/

	
	//grid_size & block_size are passed as arguments to the triple chevrons
	int numIts = 50;
	double sum3 = 0.0;
	double sum2 = 0.0;
	int sharedMemory = (block_size.x+2*KERNEL_RADIUS)*(block_size.y+2*KERNEL_RADIUS)*sizeof(float);
	QueryPerformanceFrequency(&frequency);
	for(int i = 0; i< numIts;i++)
	{
		printf("Dynamic %d\n",i);
		QueryPerformanceCounter(&t1);
		//kernelSharedCustom<<<grid_size,block_size>>>(device_array_in,device_array_out,num_elements_x,num_elements_y,device_kern);
		kernelSharedCustomDynamic<<<grid_size,block_size,sharedMemory>>>(device_array_in,device_array_out,num_elements_x,num_elements_y,device_kern,KERNEL_RADIUS);
		QueryPerformanceCounter(&t2);
		sum3 += (t2.QuadPart - t1.QuadPart) * 1000.0/ frequency.QuadPart;
	}
	cudaMemcpy(host_array2,device_array_out,num_bytes,cudaMemcpyDeviceToHost);

	for(int i = 0; i< numIts; i++)
	{
		printf("Single-Cache %d\n",i);
		QueryPerformanceCounter(&t1);
		kernelSharedCustom2<<<grid_size,block_size>>>(device_array_in2,device_array_out2,num_elements_x,num_elements_y,device_kern2);
		QueryPerformanceCounter(&t2);
		sum2 += (t2.QuadPart - t1.QuadPart) * 1000.0/ frequency.QuadPart;
	}
	
	cudaMemcpy(host_array3,device_array_out2,num_bytes,cudaMemcpyDeviceToHost);

	sum2 /= numIts;
	sum3 /= numIts;

	printf("Single-Cache processing average time: %f ms\n",sum2);
	printf("Dynamic processing average time: %f ms\n",sum3);
	
	printf("Cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));

	printf("Image downloaded from device!\n");

	FILE* fout = fopen("Custom-Dynamic-output.raw","wb");
	int written = fwrite(host_array2,sizeof(unsigned short),num_bytes/2,fout);
	fclose(fout);

	FILE* fout2 = fopen("Custom-Single-output.raw","wb");
	written = fwrite(host_array3,sizeof(unsigned short),num_bytes/2,fout2);
	fclose(fout2);
	

	printf("\n");

	//deallocate memory
	free(host_array);
	cudaFree(device_array_in);
	cudaFree(device_array_out);
	cudaFree(device_array_in2);
	cudaFree(device_array_out2);
	cudaFree(device_kern);
	cudaFree(device_kern2);
	return 0;
}