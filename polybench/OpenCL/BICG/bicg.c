/**
 * bicg.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
#define NX 4096
#define NY 4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

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
	runTimeMemInit,runTimeProgLoad, runTimeSequential,t_start, t_end,nanoSeconds0,nanoSeconds1;

char str_temp[1024];
cl_platform_id platform_id[2];
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem r_mem_obj;
cl_mem p_mem_obj;
cl_mem q_mem_obj;
cl_mem s_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<NX; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<NY; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{	
	t_start = rtclock();
	// Load the kernel source code into the array source_str
	fp = fopen("bicg.cl", "r");
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


void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r)
{	
	t_start = rtclock();
	int i, j;

  	for (i = 0; i < NX; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < NY; j++)
		{
      			A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
	
	for (i = 0; i < NY; i++)
	{
    		p[i] = i * M_PI;
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

	errcode = clGetDeviceIDs( platform_id[plataforma], plataforma ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
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


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{	
	t_start = rtclock();
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	s_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	p_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");
	
	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, r_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, r, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, s, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, p_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, p, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, q, 0, NULL, NULL);
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
		
	// Create the 1st OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "bicgKernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	// Create the 2nd OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "bicgKernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	clFinish(clCommandQue);
	t_end = rtclock();
	runTimeProgLoad = t_end - t_start;
}

void cl_launch_kernel()
{
	t_start = rtclock();	
	cl_ulong time_start;
	cl_ulong time_end;
	cl_event event0,event1;//cria evento tem q colocar no ultimo paremetro NDRANGE inicializa o evento antes no clCreateCommandQueue(CL_QUEUE_PROFILING_ENABLE)
	
	int nx=NX;
	int ny=NY;
	
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NX) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&p_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), &nx);
    errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), &ny);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the 1st OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &event0);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
	//retorno do tempo do kernel	
	clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	nanoSeconds0 = time_end-time_start;
	//até aqui
	globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&r_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&s_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), &nx);
    errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), &ny);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	
	// Execute the 2nd OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &event1);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
	//retorno do tempo do kernel
	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	nanoSeconds1 = time_end-time_start;
	//até aqui

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
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(p_mem_obj);
	errcode = clReleaseMemObject(q_mem_obj);
	errcode = clReleaseMemObject(r_mem_obj);
	errcode = clReleaseMemObject(s_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
	t_end = rtclock();
	runTimeClean = t_end - t_start;
}


void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{	
	t_start = rtclock();
	int i,j;
	
  	for (i = 0; i < NY; i++)
	{
		s[i] = 0.0;
	}

	for (i = 0; i < NX; i++)
	{
		q[i] = 0.0;
		for (j = 0; j < NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i*NY + j];
	    		q[i] = q[i] + A[i*NY + j] * p[j];
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
	fprintf(stdout,"%0.10lf\n\n",nanoSeconds1 / 1000000000.0);
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
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;
	DATA_TYPE* s_outputFromGpu;
	DATA_TYPE* q_outputFromGpu;
 	
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q_outputFromGpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	t_end = rtclock();
	runTimeDataAllocation = t_end - t_start;	

	//for(plataforma;plataforma < 2; ++plataforma){
		if(plataforma != 1)
			fprintf(stdout, "----------     GPU     ----------\n");
 		else	
			fprintf(stdout, "----------     CPU     ----------\n");		

		init_array(A, p, r);	
		read_cl_file();
		cl_initialization();
		cl_mem_init(A, r, s, p, q);
		cl_load_prog();

		cl_launch_kernel();
		
		errcode = clEnqueueReadBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0, NY*sizeof(DATA_TYPE), s_outputFromGpu, 0, NULL, NULL);
		errcode = clEnqueueReadBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0, NX*sizeof(DATA_TYPE), q_outputFromGpu, 0, NULL, NULL);
		
		if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");  
		
		cl_clean_up();
		//output runtime
		printRunTime();		
	//}		
	//bicg_cpu(A, r, s, p, q);
	printAll(); 
	
	compareResults(s, s_outputFromGpu, q, q_outputFromGpu);
	
	t_start = rtclock();
	free(A);
	free(r);
	free(s);
	free(p);
	free(q);
	free(s_outputFromGpu);
	free(q_outputFromGpu);
	t_end = rtclock();
	fprintf(stdout, "\nFree memory  runtime:	%0.10lf\n", t_end - t_start);
	fprintf(stdout, "===================================\n");
    return 0;
}

