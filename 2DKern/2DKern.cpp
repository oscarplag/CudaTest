// 2DKern.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cuda.h"
#include<cuda_runtime.h>
#include "2DKerns.cuh"

__global__ void kenel(int* temp);

int _tmain(int argc, _TCHAR* argv[])
{
	int num_elements_x = 16;
	int num_elements_y = 16;

	int num_bytes = num_elements_x*num_elements_y*sizeof(int);

	int* device_array = 0;
	int* host_array = 0;

	//allocate memory in either space
	host_array=(int*)malloc(num_bytes);
	cudaMalloc((void**)&device_array,num_bytes);

	//create two dimensional 4x4 thread blocks
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;

	//configure a two dimensional grid as well
	dim3 grid_size;
	grid_size.x = num_elements_x/block_size.x;
	grid_size.y = num_elements_y/block_size.y;

	//grid_size & block_size are passed as arguments to the triple chevrons
	kernel<<<grid_size,block_size>>>(device_array);


	//download and inspect the result on the host
	cudaMemcpy(host_array,device_array,num_bytes,cudaMemcpyDeviceToHost);

	//print out the result element by element
	for(int row = 0; row<num_elements_y; ++row)
	{
		for(int col = 0; col<num_elements_x; ++col)
		{
			printf("%2d ", host_array[row*num_elements_x+col]);
		}
		printf("\n");
	}
	printf("\n");

	//deallocate memory
	free(host_array);
	cudaFree(device_array);
	return 0;
}

