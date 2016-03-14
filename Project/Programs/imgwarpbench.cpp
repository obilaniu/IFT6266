/* Includes */
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <hdf5.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>


/* Defines */
#define OCL_INTEL_PLATFORM    "Intel"
#define OCL_INTEL_DEVICE      "Intel(R) HD Graphics Haswell GT2 Mobile"



/* Function */

/**
 * @brief mmapFile
 * @param dataPath
 * @param fdOut
 * @param fSizeOut
 * @param x_256x256Out
 * @param yOut
 * @param ptrOut
 * @return 
 */

int  mmapFile(const char* dataPath,
              int*        fdOut,
              size_t*     fSizeOut,
              size_t*     x_256x256Off,
              size_t*     yOff,
              void**      ptrOut){
	/**
	 * Open HDF5 file.
	 */
	
	hsize_t fSize;
	hid_t f = H5Fopen(dataPath, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(f<0){
		printf("Failed to open %s as HDF5 file!\n", dataPath);
		return -1;
	}else{
		H5Fget_filesize(f, &fSize);
		*fSizeOut = fSize;
		printf("Opened %llu-bytes-long HDF5 file %s ...\n",
		       fSize, dataPath);
	}
	
	
	/**
	 * Get offsets of HDF5 datasets within the file.
	 */
	
	hid_t y         = H5Dopen(f, "/data/y",         H5P_DEFAULT);
	hid_t x_64x64   = H5Dopen(f, "/data/x_64x64",   H5P_DEFAULT);
	hid_t x_128x128 = H5Dopen(f, "/data/x_128x128", H5P_DEFAULT);
	hid_t x_256x256 = H5Dopen(f, "/data/x_256x256", H5P_DEFAULT);
	if(y<0 || x_64x64<0 || x_128x128<0 || x_256x256<0){
		printf("Failed to find a dataset within the HDF5 file !\n");
		H5Dclose(y);
		H5Dclose(x_64x64);
		H5Dclose(x_128x128);
		H5Dclose(x_256x256);
		H5Fclose(f);
		return -1;
	}
	haddr_t offy         = H5Dget_offset(y);
	haddr_t offx_64x64   = H5Dget_offset(x_64x64);
	haddr_t offx_128x128 = H5Dget_offset(x_128x128);
	haddr_t offx_256x256 = H5Dget_offset(x_256x256);
	
	H5Dclose(y);
	H5Dclose(x_64x64);
	H5Dclose(x_128x128);
	H5Dclose(x_256x256);
	H5Fclose(f);
	
	*x_256x256Off = offx_256x256;
	*yOff         = offy;
	
	
	/**
	 * Open the file again, this time raw.
	 */
	
	*fdOut  = open(dataPath, O_RDONLY|O_CLOEXEC);
	if(*fdOut < 0){
		printf("Failed to open file descriptor for %s!\n", dataPath);
		return -1;
	}else{
		printf("Opened a file descriptor for %s.\n", dataPath);
	}
	
	
	/**
	 * Memory-map the file.
	 */
	
	*ptrOut = mmap(NULL,
	               fSize,
	               PROT_READ,
	               MAP_SHARED,
	               *fdOut,
	               0);
	if(*ptrOut == MAP_FAILED){
		printf("%5d: %s", errno, strerror(errno));
		printf("Failed to memory-map the file!\n");
		close(*fdOut);
		*fdOut = -1;
		return -1;
	}else{
		printf("Memory-mapped the file.\n");
	}
	
	
	/* Return */
	return 0;
}


/**
 * @brief setupOpenCL
 * @param gData
 * @return 
 */

int  setupOpenCL(const char*       kernelPath,
                 cl_platform_id*   clPID,
                 cl_device_id*     clDID,
                 cl_context*       clCtx,
                 cl_program*       clProg,
                 cl_kernel*        warpKernel,
                 cl_command_queue* cmdQ){
	/**
	 * Get # of OpenCL platforms.
	 */
	
	cl_uint NUM_PLATFORMS;
	clGetPlatformIDs(0, NULL, &NUM_PLATFORMS);
	if(NUM_PLATFORMS > 0){
		printf("Searching in %u OpenCL platform%c...\n",
		       NUM_PLATFORMS, "s"[NUM_PLATFORMS==1]);
	}else{
		printf("ERROR: Found no OpenCL platforms.\n");
		return -1;
	}
	
	
	/**
	 * Get all platform IDs for a search.
	 */
	
	cl_platform_id platformIDs[NUM_PLATFORMS];
	clGetPlatformIDs(NUM_PLATFORMS, platformIDs, NULL);
	
	
	/**
	 * Search platform IDs for the one made by "Intel".
	 * It will contain the integrated Intel GPUs.
	 */
	
	for(int i=0;i<(int)NUM_PLATFORMS;i++){
		char name[80];
		clGetPlatformInfo(platformIDs[i],
		                  CL_PLATFORM_VENDOR,
		                  sizeof(name),
		                  name,
		                  NULL);
		
		if(strcmp(name, OCL_INTEL_PLATFORM) == 0){
			*clPID = platformIDs[i];
			break;
		}
	}
	
	if(*clPID){
		printf("Found Intel OpenCL platform.\n");
	}else{
		printf("ERROR: Failed to find Intel platform!\n");
		return -1;
	}
	
	
	/**
	 * Get # of Intel GPU devices.
	 */
	
	cl_uint NUM_DEVICES;
	clGetDeviceIDs(*clPID,
	               CL_DEVICE_TYPE_GPU,
	               0,
	               NULL,
	               &NUM_DEVICES);
	if(NUM_DEVICES > 0){
		printf("Searching in %u Intel GPU device%c...\n",
		       NUM_DEVICES, "s"[NUM_DEVICES==1]);
	}else{
		printf("ERROR: Found no Intel GPU devices.\n");
		return -1;
	}
	
	
	/**
	 * Get all device IDs for a search.
	 */
	
	cl_device_id deviceIDs[NUM_DEVICES];
	clGetDeviceIDs(*clPID,
	               CL_DEVICE_TYPE_GPU,
	               NUM_DEVICES,
	               deviceIDs,
	               NULL);
	
	
	/**
	 * Find the device we want.
	 */
	
	char    devName[500];
	cl_uint devNumCUs;
	cl_uint devClockFreq;
	for(int i=0;i<(int)NUM_DEVICES;i++){
		clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME,                sizeof(devName),       devName,      NULL);
		clGetDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof(devNumCUs),    &devNumCUs,    NULL);
		clGetDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(devClockFreq), &devClockFreq, NULL);
		
		if(strncmp(devName, OCL_INTEL_DEVICE, strlen(OCL_INTEL_DEVICE)) == 0){
			*clDID = deviceIDs[i];
			break;
		}
	}
	
	if(*clDID){
		printf("Found Intel device.\n");
	}else{
		printf("ERROR: Failed to find Intel device!\n");
		return -1;
	}
	
	
	/**
	 * Print selected device.
	 */
	
	printf("Selected device is %s @ %u MHz with %u CUs\n", devName, devClockFreq, devNumCUs);
	
	
	
	/**
	 * Create OpenCL context.
	 * 
	 * We provide a single option, which might not be necessary:
	 * 
	 *     - The OpenCL platform of the context.
	 */
	
	cl_context_properties clCtxProps[] = {
	    CL_CONTEXT_PLATFORM,
	    (cl_context_properties)*clPID,
	    0
	};
	*clCtx = clCreateContext(clCtxProps,
	                         1,
	                         clDID,
	                         NULL,
	                         NULL,
	                         NULL);
	
	if(*clCtx){
		printf("Created OpenCL context.\n");
	}else{
		printf("ERROR: Failed to create OpenCL context!\n");
		return -1;
	}
	
	
	
	/**
	 * Create OpenCL program out of source code.
	 */
	
	char*       SOURCECODE;
	size_t      SOURCECODELEN;
	struct stat kernelStat;
	int kernelFd = open(kernelPath, O_RDONLY|O_CLOEXEC);
	if(kernelFd < 0){
		printf("Can't open kernel source code!\n");
		return -1;
	}
	if(fstat(kernelFd, &kernelStat)<0){
		printf("Can't stat kernel source code!\n");
		return -1;
	}
	
	SOURCECODELEN = kernelStat.st_size;
	SOURCECODE    = (char*)malloc(SOURCECODELEN+1);
	if(read(kernelFd, SOURCECODE, SOURCECODELEN) != SOURCECODELEN){
		printf("Can't read kernel source code!\n");
		return -1;
	}
	*clProg = clCreateProgramWithSource(*clCtx,
	                                    1,
	                                    (const char**)&SOURCECODE,
	                                    &SOURCECODELEN,
	                                    NULL);
	free(SOURCECODE);
	close(kernelFd);
	
	if(*clProg){
		printf("Created OpenCL program.\n");
	}else{
		printf("Error: Failed to create OpenCL program!\n");
		return -1;
	}
	
	
	
	/**
	 * Build OpenCL program.
	 */
	
	if(clBuildProgram(*clProg,
	                  0,
	                  NULL,
	                  "-cl-std=CL1.2 -cl-denorms-are-zero -cl-mad-enable -cl-fast-relaxed-math -dump-opt-asm=dump.txt",
	                  NULL,
	                  NULL) == CL_SUCCESS){
		printf("Built OpenCL program.\n");
	}else{
		printf("Error: Failed to build OpenCL program!\n");
		char buf[8192];
		clGetProgramBuildInfo(*clProg,
		                      *clDID,
		                      CL_PROGRAM_BUILD_LOG,
		                      sizeof(buf),
		                      buf,
		                      0);
		printf("***********************************************************************\n");
		printf("%s\n", buf);
		printf("***********************************************************************\n");
		return -1;
	}
	clUnloadPlatformCompiler(*clPID);
	
	
	/**
	 * Check that the warpImage and bgr2rgba kernels are present in the source code.
	 */
	
	*warpKernel = clCreateKernel(*clProg, "warpBlock4x4", NULL);
	if(*warpKernel){
		printf("Created \"warpBlock4x4\" kernel.\n");
	}else{
		printf("ERROR: Failed to create \"warpBlock4x4\" kernel!\n");
		return -1;
	}
	
	
	/**
	 * Create command queue
	 */
	
	*cmdQ = clCreateCommandQueue(*clCtx, *clDID, CL_QUEUE_PROFILING_ENABLE, NULL);
	if(*cmdQ){
		printf("Created command queue.\n");
	}else{
		printf("ERROR: Failed to create command queue!\n");
		return -1;
	}
	
	
	/**
	 * Return.
	 */
	
	return 0;
}

/**
 * Main
 */

int main(int argc, char* argv[]){
	if(argc != 3){
		printf("Usage: ./imgwarpbench path/to/catsanddogs.hdf5 path/to/warp.cl\n");
		return -1;
	}
	
	/**
	 * HDF5 memory-map business
	 */
	
	int fd;
	size_t fSize;
	size_t offx_256x256;
	size_t offy;
	void*  base;
	
	if(mmapFile(argv[1], &fd, &fSize, &offx_256x256, &offy, &base) < 0){
		printf("Failed to memory-map dataset.\n");
		return -1;
	}
	
	
	
	/**
	 * OpenCL business
	 */
	
	const int B  = 128,
	          Hi = 256,
	          Wi = 256,
	          Ho = 256,
	          Wo = 256;
	float (*hostIn)[3][Hi][Wi]  = (float(*)[3][Hi][Wi])((char*)base + offx_256x256);
	float (*hostH)[3][3]        = (float(*)[3][3])     calloc(B*3*3,     sizeof(float));
	float (*hostOut)[3][Ho][Wo] = (float(*)[3][Ho][Wo])calloc(B*3*Ho*Wo, sizeof(float));
	
	cl_platform_id   clPID     = 0;
	cl_device_id     clDID     = 0;
	cl_context       clCtx     = 0;
	cl_program       clProg    = 0;
	cl_kernel        warpKernel= 0;
	cl_command_queue cmdQ      = 0;
	if(setupOpenCL(argv[2], &clPID, &clDID, &clCtx, &clProg, &warpKernel, &cmdQ)<0){
		printf("Failed to set up OpenCL!\n");
		return -1;
	}
	cl_image_format  imgFormat = {CL_R, CL_UNSIGNED_INT8};
	cl_image_desc    imgDesc   = {
	    CL_MEM_OBJECT_IMAGE2D_ARRAY,
	    Wi, Hi, 1, 3*B,
	    Wi*sizeof(float), Hi*Wi*sizeof(float),
	    0, 0, NULL
	};
	
	
	/* Timing loop */
	for(int i=0;i<10;i++){
		/* Create fresh images */
		int err;
		cl_mem           devImg    = clCreateImage(clCtx,
		                                           CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		                                           &imgFormat,
		                                           &imgDesc,
		                                           (void*)hostIn,
		                                           &err);
		cl_mem           devH      = clCreateBuffer(clCtx,
		                                            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		                                            B*3*3*sizeof(float),
		                                            (void*)hostH,
		                                            NULL);
		cl_mem           devOut    = clCreateBuffer(clCtx,
		                                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		                                            B*3*Ho*Wo*sizeof(float),
		                                            (void*)hostOut,
		                                            NULL);
		if(!devImg){
			printf("Failed to create input image (%d).\n", err);
			return -1;
		}
		if(!devH){
			printf("Failed to create input H buffer.\n");
			return -1;
		}
		if(!devOut){
			printf("Failed to create output image buffer.\n");
			return -1;
		}
		
		/* Kernel invocation. */
		unsigned int IPERTHRD    = 1;
		unsigned int JBLKPERTHRD = 4;
		unsigned int KBLKPERTHRD = 4;
		unsigned int ROWPITCH    = Wi;
		unsigned int SLICEPITCH  = Wi*Hi;
		clSetKernelArg(warpKernel, 0, sizeof(cl_mem),       &devOut);
		clSetKernelArg(warpKernel, 1, sizeof(cl_mem),       &devImg);
		clSetKernelArg(warpKernel, 2, sizeof(cl_mem),       &devH);
		clSetKernelArg(warpKernel, 3, sizeof(unsigned int), &IPERTHRD);
		clSetKernelArg(warpKernel, 4, sizeof(unsigned int), &JBLKPERTHRD);
		clSetKernelArg(warpKernel, 5, sizeof(unsigned int), &KBLKPERTHRD);
		clSetKernelArg(warpKernel, 6, sizeof(unsigned int), &ROWPITCH);
		clSetKernelArg(warpKernel, 7, sizeof(unsigned int), &SLICEPITCH);
		
		
		struct timespec ts, te;
		clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
		
		size_t globalWSize[3] = {Wi/4/JBLKPERTHRD, Hi/4/KBLKPERTHRD, 3*B/IPERTHRD};
		//size_t localWSize[3]  = {};
		cl_event warpDoneEvt = 0;
		clEnqueueNDRangeKernel(cmdQ, warpKernel, 3,
							   NULL, globalWSize, NULL,
							   0, NULL, &warpDoneEvt);
		clWaitForEvents(1, &warpDoneEvt);
		
		clock_gettime(CLOCK_MONOTONIC_RAW, &te);
		double tsN = 1e9*ts.tv_sec + ts.tv_nsec;
		double teN = 1e9*te.tv_sec + te.tv_nsec;
		printf("Ran kernel. (%f)\n", (teN-tsN)/1e9);
		
		ulong tQueued, tSubmitted, tStarted, tEnded;
		clGetEventProfilingInfo(warpDoneEvt, CL_PROFILING_COMMAND_QUEUED,
		                        sizeof(tQueued), &tQueued, NULL);
		clGetEventProfilingInfo(warpDoneEvt, CL_PROFILING_COMMAND_SUBMIT,
		                        sizeof(tSubmitted), &tSubmitted, NULL);
		clGetEventProfilingInfo(warpDoneEvt, CL_PROFILING_COMMAND_START,
		                        sizeof(tStarted), &tStarted, NULL);
		clGetEventProfilingInfo(warpDoneEvt, CL_PROFILING_COMMAND_END,
		                        sizeof(tEnded), &tEnded, NULL);
		
		tSubmitted -= tQueued;
		tStarted   -= tQueued;
		tEnded     -= tQueued;
		
		printf("%24.9f     %24.9f     %24.9f     %24.9f\n",
		       tQueued/1e9, tSubmitted/1e9, tStarted/1e9, tEnded/1e9);
		
		clReleaseEvent(warpDoneEvt);
		
		/* Cleanup */
		clReleaseMemObject(devImg);
		clReleaseMemObject(devH);
		clReleaseMemObject(devOut);
		
		/* View results */
	}
}
