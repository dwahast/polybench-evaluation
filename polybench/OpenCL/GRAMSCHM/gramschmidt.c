/**
 * gramschmidt.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define M 2048
#define N 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef double DATA_TYPE;

int plataforma = 0;
double runTimeInitArray, runTimeReadKernel, runTimeLaunchGpuKernel,
	runTimeLaunchCpuKernel,runTimeClean,runTimeDataAllocation,runTimeKernelInit,
	runTimeMemInit,runTimeProgLoad, runTimeSequential,t_start, t_end, nanoSeconds0,nanoSeconds1,nanoSeconds2;


char str_temp[1024];
 

cl_platform_id platform_id[2];
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem r_mem_obj;
cl_mem q_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
				//printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*N + j], A_outputFromGpu[i*N + j]);
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{	
	t_start = rtclock();
	// Load the kernel source code into the array source_str
	fp = fopen("gramschmidt.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
	t_end = rtclock();
	runTimeReadKernel = t_end - t_start;
}


void init_array(DATA_TYPE* A)
{
	t_start = rtclock();
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
	t_end = rtclock();
	runTimeInitArray = t_end - t_start;
}


void cl_initialization()
{	
	t_start = rtclock();
	// Get platform and device information
	errcode = clGetPlatformIDs(2, platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id[plataforma],CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id[plataforma], CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id[plataforma], plataforma ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, 2, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, CL_QUEUE_PROFILING_ENABLE, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
	t_end = rtclock();	
	runTimeKernelInit = t_end - t_start;
}


void cl_mem_init(DATA_TYPE* A)
{	
	t_start = rtclock();
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, A, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
	t_end = rtclock();
	runTimeMemInit = t_end - t_start;
}


void cl_load_prog()
{	
	t_start = rtclock();
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "gramschmidt_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel2 = clCreateKernel(clProgram, "gramschmidt_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel3 = clCreateKernel(clProgram, "gramschmidt_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");
	clFinish(clCommandQue);
	t_end = rtclock();
	runTimeProgLoad = t_end - t_start;
}


void cl_launch_kernel()
{
	t_start = rtclock();	
	cl_ulong time_start = 0;
	cl_ulong time_end = 0;
	cl_event event0,event1,event2;
	
	int m = M;
	int n = N;

	size_t localWorkSize[2], globalWorkSizeKernel1[2], globalWorkSizeKernel2[2], globalWorkSizeKernel3[2];

	localWorkSize[0] = DIM_THREAD_BLOCK_X;
	localWorkSize[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel1[0] = DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel1[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel2[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel2[1] = 1;
	globalWorkSizeKernel3[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel3[1] = 1;
	
	int k;
	for (k = 0; k < N; k++)
	{
		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&k);
		errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&m);
		errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSizeKernel1, localWorkSize, 0, NULL, &event0);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		//clEnqueueBarrier(clCommandQue);
		clFinish(clCommandQue);
		//clWaitForEvents(1, &event0);
		clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		nanoSeconds0 += time_end-time_start;
		
		errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&k);
		errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&m);
		errcode |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSizeKernel2, localWorkSize, 0, NULL, &event1);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
		//clEnqueueBarrier(clCommandQue);
		clFinish(clCommandQue);
		//clWaitForEvents(1, &event1);
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		nanoSeconds1 += time_end-time_start;

		errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 3, sizeof(int), (void *)&k);
		errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&m);
		errcode |= clSetKernelArg(clKernel3, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 1, NULL, globalWorkSizeKernel3, localWorkSize, 0, NULL, &event2);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
		//clEnqueueBarrier(clCommandQue);		
		clFinish(clCommandQue);
		//clWaitForEvents(1, &event2);
		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		nanoSeconds2 += time_end-time_start;
	}
	clFinish(clCommandQue);
	nanoSeconds0 /= N;
	nanoSeconds1 /= N;
	nanoSeconds2 /= N;

	t_end = rtclock();
	if(plataforma != 1){
		runTimeLaunchGpuKernel = t_end - t_start;
	}else{			
		runTimeLaunchCpuKernel = t_end - t_start;
	}		
}


void cl_clean_up()
{	
	t_start = rtclock();
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(r_mem_obj);
	errcode = clReleaseMemObject(q_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
	t_end = rtclock();
	runTimeClean = t_end - t_start;
}


void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{	
	t_start = rtclock();
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i*N + k] * A[i*N + k];
		}
		
		R[k*N + k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i*N + k] = A[i*N + k] / R[k*N + k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k*N + j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k*N + j] += Q[i*N + k] * A[i*N + j];
			}
			for (i = 0; i < M; i++)
			{
				A[i*N + j] = A[i*N + j] - Q[i*N + k] * R[k*N + j];
			}
		}
	}
	t_end = rtclock();
	runTimeSequential = t_end - t_start;
}

void printRunTime(){
	fprintf(stdout, "\nInit Array 	 runtime:%0.10lf\n", runTimeInitArray);
	fprintf(stdout, "Read Kernel  runtime:%0.10lf\n", runTimeReadKernel);
	fprintf(stdout, "Kernels Init runtime:%0.10lf\n", runTimeKernelInit);
	fprintf(stdout, "Memory Init  runtime:%0.10lf\n", runTimeMemInit);
	fprintf(stdout, "Prog Load    runtime:%0.10lf\n", runTimeProgLoad);
	//kernels individuais
	fprintf(stdout,"\nKernels 	 runtime:");
	fprintf(stdout,"%0.10lf:",nanoSeconds0 / 1000000000.0);
	fprintf(stdout,"%0.10lf:",nanoSeconds1 / 1000000000.0);
	fprintf(stdout,"%0.10lf\n\n",nanoSeconds2 / 1000000000.0);
	//
	if(plataforma != 1){
		fprintf(stdout, "Launch Kernl runtime:%0.10lf\n", runTimeLaunchGpuKernel);
	}else{			
		fprintf(stdout, "Launch Kernl runtime:%0.10lf\n", runTimeLaunchCpuKernel);
	}		
	fprintf(stdout, "Clean  up    runtime:%0.10lf\n\n", runTimeClean);	
}
void printAll(){
	fprintf(stdout, "---------Sequential CPU----------\n\n");
	fprintf(stdout, "Sequential 	 runtime:%0.10lf\n\n", runTimeSequential);
	fprintf(stdout, "----------     ALL     ----------\n\n");
	fprintf(stdout, "CPU Kernel 	 runtime:%0.10lf\n", runTimeLaunchCpuKernel);
	fprintf(stdout, "GPU Kernel 	 runtime:%0.10lf\n", runTimeLaunchGpuKernel);
	fprintf(stdout, "Sequential 	 runtime:%0.10lf\n", runTimeSequential);
	fprintf(stdout, "\nData Alloc   runtime:%0.10lf\n\n", runTimeDataAllocation);
}

int main(int argc, char const *argv[])
{	
	plataforma = atoi(argv[1]);	
	t_start = rtclock();
	DATA_TYPE* A;
	DATA_TYPE* A_outputFromGpu;
	DATA_TYPE* R;
	DATA_TYPE* Q;
	
	A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	A_outputFromGpu = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  

	t_end = rtclock();
	runTimeDataAllocation = t_end - t_start;

	//for(plataforma; plataforma < 2; ++plataforma){
		if(plataforma != 1)
			fprintf(stdout, "----------     GPU     ----------\n");
 		else	
			fprintf(stdout, "----------     CPU     ----------\n");			

		init_array(A);
		read_cl_file();
		cl_initialization();
		cl_mem_init(A);
		cl_load_prog();

		cl_launch_kernel();

		errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, M*N*sizeof(DATA_TYPE), A_outputFromGpu, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");   

		cl_clean_up();
		//output runtime
		printRunTime();
	//}	
	//gramschmidt(A, R, Q);
	printAll(); 	
	
	compareResults(A, A_outputFromGpu);
	
	
	t_start = rtclock();
	free(A);
	free(A_outputFromGpu);
	free(R);
	free(Q);  
	t_end = rtclock();
	fprintf(stdout, "\nFree memory  runtime:	%0.10lf\n", t_end - t_start);
	fprintf(stdout, "===================================\n");
	return 0;
}

