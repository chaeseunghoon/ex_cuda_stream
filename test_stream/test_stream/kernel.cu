
#define CUDA_API_PER_THREAD_DEFAULT_STEAM

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

__global__ void warm_up_gpu() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

bool load_image(char* f_name, unsigned short *img, size_t mem_size)
{
	FILE* fp;

	fopen_s(&fp, f_name, "rb");
	fread(img, sizeof(unsigned short), mem_size, fp);
	fclose(fp);

	return true;
}

bool write_image(char* f_name, float* img, size_t mem_size)
{
	FILE* fp;

	fopen_s(&fp, f_name, "wb");
	fwrite(img, sizeof(float), mem_size, fp);
	fclose(fp);

	return true;
}


__global__ void log_kernel(float *dst, unsigned short* src, int w, int h)
{
	unsigned long count = w * h;
	size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	float logair = __log10f(65535.f);

	if (idx_x < w && idx_y < h )
	{
		if (src[idx_y * w + idx_x] != 0)
			dst[idx_y * w + idx_x] = __log10f(65535.f / src[idx_y * w + idx_x]);
		else
			dst[idx_y * w + idx_x] = logair;
	}

}

void null_stream(void)
{
	char buf_file[256];
	unsigned short* img = NULL;
	unsigned short* d_src;
	float* d_dst;
	float* output;

	int i;
	int w = 2048;
	int h = 2048;
	clock_t start, end;
	float res;
	size_t mem_size = static_cast<size_t>(w) * h;
	cudaError_t error;

	int nx = 32;
	int ny = 32;
	dim3 dimLogGrid(ceil(float(w) / nx), ceil(float(h) / ny));
	dim3 dimLogBlock(nx, ny);

	printf("warm_up_gpu start\n");
	warm_up_gpu << < 1, 1 >> > ();
	printf("warm_up_gpu end\n");

	start = clock();

	cudaMallocHost(&img, sizeof(unsigned short) * mem_size);
	cudaMallocHost(&output, sizeof(float) * mem_size);

	cudaMalloc(&d_src, sizeof(unsigned short) * mem_size);
	cudaMalloc(&d_dst, sizeof(float) * mem_size);

	for (i = 0; i < 360; i++)
	{
		sprintf(buf_file, "E:\\sample\\projections2\\ViewImage%04d.raw", i);
		load_image(buf_file, img, mem_size);
		cudaMemcpy(d_src, img, sizeof(unsigned short) * mem_size, cudaMemcpyHostToDevice);
		log_kernel << <dimLogGrid, dimLogBlock >> > (d_dst, d_src, 2048, 2048);

		error = cudaMemcpy(output, d_dst, sizeof(float) * mem_size, cudaMemcpyDeviceToHost);

		sprintf(buf_file, "E:\\sample\\output\\ViewImage%04d.raw", i);
		write_image(buf_file, output, mem_size);
	}
	

	end = clock();
	res = (float)(end - start) / CLOCKS_PER_SEC;
	printf("run-time : %.3f s\n", res);

	cudaFreeHost(img);
	cudaFreeHost(output);
	cudaFree(d_src);
	cudaFree(d_dst);


}

void non_null_stream(void)
{
	/*
	* device가 비동기식으로 동작하여 바로 return 을 수행
	* device의 stream간은 비동기식으로 잘 동작하지만, 바로 return되기 때문에 host에서 문제가 발생
	* 파일 로딩 -> device stream 1 수행 -> device stream 1 완료되기 전에 파일 저장(파일이 제대로 저장되지 않음)
	* num_streams 만큼 수행 하면 output_s 변수에 그전 결과가 저장되어 있기 때문에 첫 num_streams 를 제외하고는 제대로 수행되는것처럼 보이지만 동기화 문제로 잘못 수행된 상태임
	*/
	char buf_file[256];

	unsigned short* img_s = NULL;
	unsigned short* d_src_s;
	float* d_dst_s;
	float* output_s;

	int num_streams = 4; // 2의 배수

	int i, j;
	int w = 2048;
	int h = 2048;
	clock_t start, end;
	float res;
	size_t mem_size = static_cast<size_t>(w) * h;

	int nx = 32;
	int ny = 32;
	dim3 dimLogGrid(ceil(float(w) / nx), ceil(float(h) / ny));
	dim3 dimLogBlock(nx, ny);
	cudaStream_t* stream;

	printf("warm_up_gpu start\n");
	warm_up_gpu << < 1, 1 >> > ();
	printf("warm_up_gpu end\n");

	start = clock();
	cudaMallocHost(&stream, sizeof(cudaStream_t) * num_streams);

	cudaMallocHost(&img_s, num_streams * sizeof(unsigned short) * mem_size);
	cudaMallocHost(&output_s, num_streams * sizeof(float) * mem_size);

	cudaMalloc(&d_src_s, num_streams * sizeof(unsigned short) * mem_size);
	cudaMalloc(&d_dst_s, num_streams * sizeof(float) * mem_size);

	cudaError_t error;

	for (i = 0; i < num_streams; i++)
		cudaStreamCreate(&stream[i]);

	for (i = 0; i < (360 / num_streams); i++)
	{
		for (j = 0; j < num_streams; j++)
		{
			sprintf(buf_file, "E:\\9860A_2\\projections2\\ViewImage%04d.raw", num_streams * i + j);
			load_image(buf_file, &img_s[j * mem_size], mem_size);

			cudaMemcpyAsync(&d_src_s[j * mem_size], &img_s[j * mem_size], sizeof(unsigned short) * mem_size, cudaMemcpyHostToDevice, stream[j]);
			log_kernel << <dimLogGrid, dimLogBlock, 0, stream[j] >> > (&d_dst_s[j * mem_size], &d_src_s[j * mem_size], 2048, 2048);

			error = cudaMemcpyAsync(&output_s[j * mem_size], &d_dst_s[j * mem_size], sizeof(float) * mem_size, cudaMemcpyDeviceToHost, stream[j]);

			if (i == 0)
				cudaStreamSynchronize(stream[j]);

			error = cudaStreamQuery(stream[j]);

			sprintf(buf_file, "E:\\9860A_2\\output\\ViewImage%04d.raw", num_streams * i + j);
			write_image(buf_file, &output_s[j * mem_size], mem_size);

		}

	}

	for (i = 0; i < num_streams; i++)
		cudaStreamDestroy(stream[i]);

	end = clock();
	res = (float)(end - start) / CLOCKS_PER_SEC;
	printf("run-time : %.3f s\n", res);

	cudaFreeHost(img_s);
	cudaFreeHost(output_s);
	cudaFree(d_src_s);
	cudaFree(d_dst_s);
}

void openmp_non_null_stream(void)
{


	unsigned short* img_s = NULL;
	unsigned short* d_src_s;
	float* d_dst_s;
	float* output_s;

	int num_streams = 8; // 2의 배수

	int i, j;
	int w = 2048;
	int h = 2048;
	clock_t start, end;
	float res;
	size_t mem_size = static_cast<size_t>(w) * h;
	int num = 360;

	int nx = 32;
	int ny = 32;
	dim3 dimLogGrid(ceil(float(w) / nx), ceil(float(h) / ny));
	dim3 dimLogBlock(nx, ny);
	cudaStream_t* stream;
	printf("openmp_non_null_stream\n");

	start = clock();
	printf("warm_up_gpu start\n");
	warm_up_gpu << < 1, 1 >> > ();
	printf("warm_up_gpu end\n");
	i = 0;
	cudaMallocHost(&stream, sizeof(cudaStream_t) * num_streams);		

	cudaMallocHost(&img_s, num_streams * sizeof(unsigned short) * mem_size);
	cudaMallocHost(&output_s, num_streams * sizeof(float) * mem_size);

	cudaMalloc(&d_src_s, num_streams * sizeof(unsigned short) * mem_size);
	cudaMalloc(&d_dst_s, num_streams * sizeof(float) * mem_size);

// 프로젝트 속성 -> CUDA C/C++ -> Host -> Additional compiler Options -> /openmp 추가
#pragma omp parallel for
	for (j = 0; j < num_streams; j++)
	{
		char buf_file[256];
		int num_task;

		cudaStreamCreate(&stream[j]);

		while (true)
		{
#pragma omp critical(i)
			{
				num_task = i;
				i++;
			}

			if (num_task >= num)
			{

				break;
			}

			sprintf(buf_file, "E:\\9860A_2\\projections2\\ViewImage%04d.raw", num_task);
			load_image(buf_file, &img_s[j * mem_size], mem_size);

			cudaMemcpyAsync(&d_src_s[j * mem_size], &img_s[j * mem_size], sizeof(unsigned short) * mem_size, cudaMemcpyHostToDevice, stream[j]);
			log_kernel << <dimLogGrid, dimLogBlock, 0, stream[j] >> > (&d_dst_s[j * mem_size], &d_src_s[j * mem_size], 2048, 2048);

			cudaMemcpyAsync(&output_s[j * mem_size], &d_dst_s[j * mem_size], sizeof(float) * mem_size, cudaMemcpyDeviceToHost, stream[j]);

			cudaStreamSynchronize(stream[j]);

			sprintf(buf_file, "E:\\9860A_2\\output\\ViewImage%04d.raw", num_task);
			write_image(buf_file, &output_s[j * mem_size], mem_size);
		}
	}
	
	end = clock();
	res = (float)(end - start) / CLOCKS_PER_SEC;
	printf("run-time : %.3f s\n", res);

	for (i = 0; i < num_streams; i++)
		cudaStreamDestroy(stream[i]);
	cudaFreeHost(stream);
	cudaFreeHost(img_s);
	cudaFreeHost(output_s);
	cudaFree(d_src_s);
	cudaFree(d_dst_s);
}

void openmp_null_stream(void)
{


	unsigned short* img_s = NULL;
	unsigned short* d_src_s;
	float* d_dst_s;
	float* output_s;

	int num_streams = 8; // 2의 배수

	int i, j;
	int w = 2048;
	int h = 2048;
	clock_t start, end;
	float res;
	size_t mem_size = static_cast<size_t>(w) * h;
	int num = 360;

	int nx = 32;
	int ny = 32;
	dim3 dimLogGrid(ceil(float(w) / nx), ceil(float(h) / ny));
	dim3 dimLogBlock(nx, ny);
	cudaStream_t stream;
	printf("openmp_null_stream");

	start = clock();
	printf("warm_up_gpu start\n");
	warm_up_gpu << < 1, 1 >> > ();
	printf("warm_up_gpu end\n");
	i = 0;
	
	cudaMallocHost(&img_s, num_streams * sizeof(unsigned short) * mem_size);
	cudaMallocHost(&output_s, num_streams * sizeof(float) * mem_size);

	cudaMalloc(&d_src_s, num_streams * sizeof(unsigned short) * mem_size);
	cudaMalloc(&d_dst_s, num_streams * sizeof(float) * mem_size);

	// 프로젝트 속성 -> CUDA C/C++ -> Host -> Additional compiler Options -> /openmp 추가
#pragma omp parallel for
	for (j = 0; j < num_streams; j++)
	{
		char buf_file[256];
		int num_task;

		while (true)
		{
#pragma omp critical(i)
			{
				num_task = i;
				i++;
			}

			if (num_task >= num)
			{
				break;
			}

			sprintf(buf_file, "E:\\9860A_2\\projections2\\ViewImage%04d.raw", num_task);
			load_image(buf_file, &img_s[j * mem_size], mem_size);

			cudaMemcpyAsync(&d_src_s[j * mem_size], &img_s[j * mem_size], sizeof(unsigned short) * mem_size, cudaMemcpyHostToDevice);
			log_kernel << <dimLogGrid, dimLogBlock >> > (&d_dst_s[j * mem_size], &d_src_s[j * mem_size], 2048, 2048);

			cudaMemcpyAsync(&output_s[j * mem_size], &d_dst_s[j * mem_size], sizeof(float) * mem_size, cudaMemcpyDeviceToHost);

			//cudaStreamSynchronize(stream[j]);

			sprintf(buf_file, "E:\\9860A_2\\output\\ViewImage%04d.raw", num_task);
			write_image(buf_file, &output_s[j * mem_size], mem_size);
		}
	}

	end = clock();
	res = (float)(end - start) / CLOCKS_PER_SEC;
	printf("run-time : %.3f s\n", res);

	cudaFreeHost(img_s);
	cudaFreeHost(output_s);
	cudaFree(d_src_s);
	cudaFree(d_dst_s);
}

int main(int argc, char* argv[])
{
	int mode = 3;

	if (mode == 0)
		null_stream();
	else if (mode == 1)
		non_null_stream();
	else if (mode == 2)
		openmp_non_null_stream();
	else if (mode == 3)
		openmp_null_stream();

	return 0;
}