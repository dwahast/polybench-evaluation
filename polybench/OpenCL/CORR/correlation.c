/**
 * correlation.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define M 2048
#define N 2048

/* Thread block dimensions for kernel 1*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_X 32
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_Y 8

/* Thread block dimensions for kernel 4*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_4_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_4_Y 1


#define sqrt_of_array_cell(x,j) sqrt(x[j])

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
	runTimeMemInit,runTimeProgLoad, runTimeSequential,t_start, t_end, nanoSeconds0, nanoSeconds1, nanoSeconds2, nanoSeconds3;

char str_temp[1024];

#define FLOAT_N 3214212.01
#define EPS 0.005

cl_platform_id platform_id[2];
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_std;
cl_kernel clKernel_reduce;
cl_kernel clKernel_corr;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem stddev_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i<=M; i++)
	{
		for (j=0; j<=N; j++)
		{
			if (percentDiff(symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
				//printf("I: %d J: %d \n 1: %f\n 2: %f\n", i, j, symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]);		
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{	
	t_start = rtclock();
	// Load the kernel source code into the array source_str
	fp = fopen("correlation.cl", "r");
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


void init_arrays(DATA_TYPE* data)
{	
	t_start = rtclock();
	int i, j;
	
	for (i=0; i<=M; i++) 
	{
    	for (j=0; j<=N; j++) 
		{
       		data[i*N + j] = ((DATA_TYPE) i*j)/ (M+1);	
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


void cl_mem_init(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat)
{	
	t_start = rtclock();
	data_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &errcode);
	symmat_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &errcode);
	stddev_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1), NULL, &errcode);
	mean_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1), NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), data, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), symmat, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, stddev_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1), stddev, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1), mean, 0, NULL, NULL);
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
	clKernel_mean = clCreateKernel(clProgram, "mean_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel_std = clCreateKernel(clProgram, "std_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel_reduce = clCreateKernel(clProgram, "reduce_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");

	clKernel_corr = clCreateKernel(clProgram, "corr_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel4\n");
	clFinish(clCommandQue);
	t_end = rtclock();
	runTimeProgLoad = t_end - t_start;
}


void cl_launch_kernel()
{
	t_start = rtclock();	
	cl_ulong time_start;
	cl_ulong time_end;
	cl_event event0,event1,event2,event3;

	int m = M;
	int n = N;

	DATA_TYPE float_n = FLOAT_N;
	DATA_TYPE eps = EPS;

	size_t localWorkSize_Kernel1[2], globalWorkSize_Kernel1[2];
	size_t localWorkSize_Kernel2[2], globalWorkSize_Kernel2[2];
	size_t localWorkSize_Kernel3[2], globalWorkSize_Kernel3[2];
	size_t localWorkSize_Kernel4[2], globalWorkSize_Kernel4[2];

	localWorkSize_Kernel1[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	localWorkSize_Kernel1[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	globalWorkSize_Kernel1[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	globalWorkSize_Kernel1[1] = 1;

	localWorkSize_Kernel2[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	localWorkSize_Kernel2[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
	globalWorkSize_Kernel2[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	globalWorkSize_Kernel2[1] = 1;

	localWorkSize_Kernel3[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	localWorkSize_Kernel3[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
	globalWorkSize_Kernel3[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	globalWorkSize_Kernel3[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

	localWorkSize_Kernel4[0] = DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	localWorkSize_Kernel4[1] = DIM_LOCAL_WORK_GROUP_KERNEL_4_Y;
	globalWorkSize_Kernel4[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_4_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	globalWorkSize_Kernel4[1] = 1;


	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_mean, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_mean, 3, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_mean, 4, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, &event0);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	//clEnqueueBarrier(clCommandQue);
	clFinish(clCommandQue);
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_std, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  clSetKernelArg(clKernel_std, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= clSetKernelArg(clKernel_std, 2, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_std, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_std, 4, sizeof(DATA_TYPE), (void *)&eps);
	errcode |= clSetKernelArg(clKernel_std, 5, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_std, 6, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");
 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_std, 1, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, &event1);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	//clEnqueueBarrier(clCommandQue);
	clFinish(clCommandQue);
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_reduce, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  clSetKernelArg(clKernel_reduce, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= clSetKernelArg(clKernel_reduce, 2, sizeof(cl_mem), (void *)&data_mem_obj);	
	errcode |= clSetKernelArg(clKernel_reduce, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_reduce, 4, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_reduce, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments3\n");
 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, &event2);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
	//clEnqueueBarrier(clCommandQue);
	clFinish(clCommandQue);
	// Set the arguments of the kernel	
	errcode =  clSetKernelArg(clKernel_corr, 0, sizeof(cl_mem), (void *)&symmat_mem_obj);
	errcode |= clSetKernelArg(clKernel_corr, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_corr, 2, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_corr, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments4\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_corr, 1, NULL, globalWorkSize_Kernel4, localWorkSize_Kernel4, 0, NULL, &event3);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel4\n");
	//clEnqueueBarrier(clCommandQue);
	clFinish(clCommandQue);
	DATA_TYPE val = 1.0;
	clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, ((M)*(M+1) + (M))*sizeof(DATA_TYPE), sizeof(DATA_TYPE), &val, 0, NULL, NULL);

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
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	nanoSeconds3 = time_end-time_start;


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
	errcode = clReleaseKernel(clKernel_reduce);
	errcode = clReleaseKernel(clKernel_mean);
	errcode = clReleaseKernel(clKernel_std);
	errcode = clReleaseKernel(clKernel_corr);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(symmat_mem_obj);
	errcode = clReleaseMemObject(data_mem_obj);
	errcode = clReleaseMemObject(mean_mem_obj);
	errcode = clReleaseMemObject(stddev_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
	t_end = rtclock();
	runTimeClean = t_end - t_start;
}


void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat)
{	
	t_start = rtclock();
	int i, j, j1, j2;	
	
	// Determine mean of column vectors of input data matrix 
  	for (j = 1; j <= M; j++)
   	{
  		mean[j] = 0.0;

   		for (i = 1; i <= N; i++)
		{
			mean[j] += data[i*(M+1) + j];
   		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
   	}

	// Determine standard deviations of column vectors of data matrix. 
  	for (j = 1; j <= M; j++)
   	{
   		stddev[j] = 0.0;
      
		for (i = 1; i <= N; i++)
		{
			stddev[j] += (data[i*(M+1) + j] - mean[j]) * (data[i*(M+1) + j] - mean[j]);
		}
		
		stddev[j] /= FLOAT_N;
    	stddev[j] = sqrt_of_array_cell(stddev, j);
    	stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
    }

 	// Center and reduce the column vectors. 
  	for (i = 1; i <= N; i++)
	{
    	for (j = 1; j <= M; j++)
    	{
			data[i*(M+1) + j] -= mean[j];
			data[i*(M+1) + j] /= sqrt(FLOAT_N) ;
			data[i*(M+1) + j] /= stddev[j];
    	}
	}

	// Calculate the m * m correlation matrix. 
  	for (j1 = 1; j1 <= M-1; j1++)
    {	
    	symmat[j1*(M+1) + j1] = 1.0;
    
		for (j2 = j1+1; j2 <= M; j2++)
		{
	  		symmat[j1*(M+1) + j2] = 0.0;

	  		for (i = 1; i <= N; i++)
			{
	   			symmat[j1*(M+1) + j2] += (data[i*(M+1) + j1] * data[i*(M+1) + j2]);
			}

	  		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
		}
    }
 
	symmat[M*(M+1) + M] = 1.0;

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
	fprintf(stdout,"%0.10lf:",nanoSeconds2 / 1000000000.0);
	fprintf(stdout,"%0.10lf\n\n",nanoSeconds3 / 1000000000.0);
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
	DATA_TYPE* data;
	DATA_TYPE* mean;
	DATA_TYPE* stddev;
	DATA_TYPE* symmat;
	DATA_TYPE* symmat_outputFromGpu;

	data = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
	mean = (DATA_TYPE*)malloc((M + 1)*sizeof(DATA_TYPE));
	stddev = (DATA_TYPE*)malloc((M + 1)*sizeof(DATA_TYPE));
	symmat = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
	symmat_outputFromGpu = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
	t_end = rtclock();
	runTimeDataAllocation = t_end - t_start;
	for(plataforma; plataforma < 2; ++plataforma){
		if(plataforma != 1)
			fprintf(stdout, "----------     GPU     ----------\n");
 		else	
			fprintf(stdout, "----------     CPU     ----------\n");			

		init_arrays(data);
		read_cl_file();
		cl_initialization();
		cl_mem_init(data, mean, stddev, symmat);
		cl_load_prog();

		cl_launch_kernel();

		errcode = clEnqueueReadBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, (M+1) * (N+1) * sizeof(DATA_TYPE), symmat_outputFromGpu, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

		cl_clean_up();
		//output runtime
		printRunTime();
	}	
	//correlation(data, mean, stddev, symmat);
	printAll(); 	
	   
	compareResults(symmat, symmat_outputFromGpu);
	
	t_start = rtclock();
	free(data);
	free(mean);
	free(stddev);
	free(symmat);
	free(symmat_outputFromGpu);
	t_end = rtclock();
	fprintf(stdout, "\nFree memory  runtime:	%0.10lf\n", t_end - t_start);
	fprintf(stdout, "===================================\n");
    return 0;
}

