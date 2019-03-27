/**
 * fdtd2d.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

/* Problem size */
#define TMAX 500
#define NX 2048
#define NY 2048

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

int plataforma = 0;
double runTimeInitArray, runTimeReadKernel, runTimeLaunchGpuKernel,
	runTimeLaunchCpuKernel,runTimeClean,runTimeDataAllocation,runTimeKernelInit,
	runTimeMemInit,runTimeProgLoad, runTimeSequential,t_start, t_end, nanoSeconds0, nanoSeconds1, nanoSeconds2;

char str_temp[1024];

DATA_TYPE alpha = 23;
DATA_TYPE beta = 15;

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

cl_mem fict_mem_obj;
cl_mem ex_mem_obj;
cl_mem ey_mem_obj;
cl_mem hz_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
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
	fp = fopen("fdtd2d.cl", "r");
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


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{	
	t_start = rtclock();
	int i, j;

  	for (i = 0; i < TMAX; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
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


void cl_mem_init(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{	
	t_start = rtclock();
	fict_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * TMAX, NULL, &errcode);
	ex_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * (NY + 1), NULL, &errcode);
	ey_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (NX + 1) * NY, NULL, &errcode);
	hz_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, fict_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * TMAX, _fict_, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, ex_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * (NY + 1), ex, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, ey_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (NX + 1) * NY, ey, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, hz_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, hz, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "fdtd_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	
	// Create the OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "fdtd_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	// Create the OpenCL kernel
	clKernel3 = clCreateKernel(clProgram, "fdtd_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
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

	int nx = NX;
	int ny = NY;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NX) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;


	int t;
	for(t=0;t<TMAX;t++)
	{
		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&fict_mem_obj);
		errcode =  clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&ex_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&ey_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&hz_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&t);
		errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&nx);
		errcode |= clSetKernelArg(clKernel1, 6, sizeof(int), (void *)&ny);
		
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event0);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		//clEnqueueBarrier(clCommandQue);
		clFinish(clCommandQue);
		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&nx);
		errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&ny);
		
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event1);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		//clEnqueueBarrier(clCommandQue);
		clFinish(clCommandQue);
		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 3, sizeof(int), (void *)&nx);
		errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&ny);
		
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event2);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		clFinish(clCommandQue);

		clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		nanoSeconds0 = time_end-time_start;
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		nanoSeconds1 = time_end-time_start;
		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		nanoSeconds2 = time_end-time_start;

	}

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
	errcode = clReleaseMemObject(fict_mem_obj);
	errcode = clReleaseMemObject(ex_mem_obj);
	errcode = clReleaseMemObject(ey_mem_obj);
	errcode = clReleaseMemObject(hz_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
	t_end = rtclock();
	runTimeClean = t_end - t_start;
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{	
	t_start = rtclock();
	int t, i, j;
	
	for(t=0; t < TMAX; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       			for (j = 0; j < NY; j++)
				{
       				ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
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
	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	DATA_TYPE* hz_outputFromGpu;

	_fict_ = (DATA_TYPE*)malloc(TMAX*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

	t_end = rtclock();
	runTimeDataAllocation = t_end - t_start;

	//for(plataforma; plataforma < 2; ++plataforma){
		if(plataforma != 1)
			fprintf(stdout, "----------     GPU     ----------\n");
 		else	
			fprintf(stdout, "----------     CPU     ----------\n");			

		int i;
		init_arrays(_fict_, ex, ey, hz);
		read_cl_file();
		cl_initialization();
		cl_mem_init(_fict_, ex, ey, hz);
		cl_load_prog();

		cl_launch_kernel();

		errcode = clEnqueueReadBuffer(clCommandQue, hz_mem_obj, CL_TRUE, 0, NX * NY * sizeof(DATA_TYPE), hz_outputFromGpu, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");	
		
		cl_clean_up();
		//output runtime
		printRunTime();
	//}
	//runFdtd(_fict_, ex, ey, hz);
	printAll(); 	   

	compareResults(hz, hz_outputFromGpu);
	
	t_start = rtclock();
	free(_fict_);
	free(ex);
	free(ey);
	free(hz);
	free(hz_outputFromGpu);
	t_end = rtclock();
	fprintf(stdout, "\nFree memory  runtime:	%0.10lf\n", t_end - t_start);
	fprintf(stdout, "===================================\n");
    return 0;
}

