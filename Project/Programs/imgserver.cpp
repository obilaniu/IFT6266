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
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>


/* Defines */
#define OCL_INTEL_PLATFORM    "Intel"
#define OCL_INTEL_DEVICE      "Intel(R) HD Graphics Haswell GT2 Mobile"



/* Data structures */

/**
 * Global OpenCL data.
 */

typedef struct{
	cl_platform_id intelPlatformID;
	cl_device_id   intelDeviceID;
	cl_context     context;
	cl_program     program;
	cl_kernel      kernel;
} OCL_DATA;

/**
 * Server Socket Data
 */

typedef struct{
	int   fd;
} SSOCK_DATA;

/**
 * Per-connection context.
 */

typedef struct{
	cl_command_queue cmdQ;
	cl_kernel        kernel;
} CONN_CTX;



/* Functions */

/********************* Global OpenCL state code ***********************/

/**
 * Load OpenCL source code.
 */

void oclLoadSourceCode(const char** SOURCECODE, size_t* SOURCECODELEN){
	*SOURCECODE = 
	"kernel void warpImage(__global float*           dst,\n"
	"                      read_only image2d_array_t src,\n"
	"                      __constant float*         Harray){\n"
	"    /* Get IDs. */\n"
	"    size_t x     =  get_global_id(0);\n"
	"    size_t y     =  get_global_id(1);\n"
	"    size_t b     =  get_global_id(2);\n"
	"    size_t sX    =  get_global_size(0);\n"
	"    size_t sY    =  get_global_size(1);\n"
	"    size_t sB    =  get_global_size(2);\n"
	"    \n"
	"    \n"
	"    /* Apply transformation matrix. */\n"
	"    __constant float(*H)[3] = (__constant float(*)[3])&Harray[b*3*3];\n"
	"    float3 row0  = (H[0][0],H[0][1],H[0][2]);\n"
	"    float3 row1  = (H[1][0],H[1][1],H[1][2]);\n"
	"    float3 row2  = (H[2][0],H[2][1],H[2][2]);\n"
	"    float3 v     = (x,y,1);\n"
	"    \n"
	"    float  xp    = dot(row0, v);\n"
	"    float  yp    = dot(row1, v);\n"
	"    float  zp    = dot(row2, v);\n"
	"    \n"
	"    float4 coord = (float4)(xp/zp,yp/zp,b,0.0);\n"
	"    \n"
	"    \n"
	"    /**\n"
	"     * Sample w/ linear interp from the image texture.\n"
	"     *\n"
	"     * Since the image is actually BGR, we also swap.\n"
	"     */\n"
	"    \n"
	"    const sampler_t smpl = CLK_NORMALIZED_COORDS_TRUE |\n"
	"                           CLK_ADDRESS_CLAMP          |\n"
	"                           CLK_FILTER_LINEAR          ;\n"
	"    float3 rgb = read_imagef(src, smpl, coord).zyx;\n"
	"    \n"
	"    \n"
	"    /* Write out the data. */\n"
	"    size_t planeOff = sX*sY;\n"
	"    size_t baseOff  = b*3*planeOff;\n"
	"    size_t pixOff   = y*sX + x;\n"
	"    \n"
	"    dst[baseOff + 0*planeOff + pixOff] = rgb.x;\n"
	"    dst[baseOff + 1*planeOff + pixOff] = rgb.y;\n"
	"    dst[baseOff + 2*planeOff + pixOff] = rgb.z;\n"
	"}\n";
	*SOURCECODELEN = strlen(*SOURCECODE);
}

/**
 * OpenCL setup.
 */

int  oclSetup(OCL_DATA* oclData){
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
	oclData->intelPlatformID = NULL;
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
			oclData->intelPlatformID = platformIDs[i];
			break;
		}
	}
	
	if(oclData->intelPlatformID){
		printf("Found Intel OpenCL platform.\n");
	}else{
		printf("ERROR: Failed to find Intel platform!\n");
		return -1;
	}
	
	
	/**
	 * Get # of Intel GPU devices.
	 */
	
	cl_uint NUM_DEVICES;
	clGetDeviceIDs(oclData->intelPlatformID,
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
	oclData->intelDeviceID = NULL;
	clGetDeviceIDs(oclData->intelPlatformID,
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
			oclData->intelDeviceID = deviceIDs[i];
			break;
		}
	}
	
	if(oclData->intelDeviceID){
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
	 * Create context.
	 */
	
	cl_context_properties clCtxProp[] = {
	    CL_CONTEXT_PLATFORM,
	    (cl_context_properties)oclData->intelPlatformID,
	    0
	};
	oclData->context = clCreateContext(clCtxProp,
	                                   1,
	                                   &oclData->intelDeviceID,
	                                   NULL,
	                                   NULL,
	                                   NULL);
	
	if(oclData->context){
		printf("Created OpenCL context.\n");
	}else{
		printf("ERROR: Failed to create OpenCL context!\n");
		clReleaseDevice(oclData->intelDeviceID);
		return -1;
	}
	
	/**
	 * Create program out of source code.
	 */
	
	const char* SOURCECODE;
	size_t      SOURCECODELEN;
	oclLoadSourceCode(&SOURCECODE, &SOURCECODELEN);
	oclData->program = clCreateProgramWithSource(oclData->context,
	                                             1,
	                                             &SOURCECODE,
	                                             &SOURCECODELEN,
	                                             NULL);
	//free(SOURCECODE);
	
	if(oclData->program){
		printf("Created OpenCL program.\n");
	}else{
		printf("Error: Failed to create OpenCL program!\n");
		clReleaseDevice(oclData->intelDeviceID);
		clReleaseContext(oclData->context);
		return -1;
	}
	
	
	/**
	 * Build program.
	 */
	
	if(clBuildProgram(oclData->program,
	                  0,
	                  NULL,
	                  "-cl-std=CL1.2 -cl-denorms-are-zero -cl-mad-enable -cl-fast-relaxed-math",
	                  NULL,
	                  NULL) == CL_SUCCESS){
		printf("Built OpenCL program.\n");
	}else{
		printf("Error: Failed to build OpenCL program!\n");
		char buf[8192];
		clGetProgramBuildInfo(oclData->program,
		                      oclData->intelDeviceID,
		                      CL_PROGRAM_BUILD_LOG,
		                      sizeof(buf),
		                      buf,
		                      0);
		printf("***********************************************************************\n");
		printf("%s\n", buf);
		printf("***********************************************************************\n");
		clReleaseProgram(oclData->program);
		clReleaseDevice(oclData->intelDeviceID);
		clReleaseContext(oclData->context);
		return -1;
	}
	clUnloadPlatformCompiler(oclData->intelPlatformID);
	
	
	/**
	 * Get a hold of the warpImage kernel.
	 */
	
	cl_kernel kernel = clCreateKernel(oclData->program, "warpImage", NULL);
	if(kernel){
		printf("Created \"warpImage\" kernel.\n");
		clReleaseKernel(kernel);
	}else{
		printf("ERROR: Failed to create \"warpImage\" kernel!\n");
		clReleaseProgram(oclData->program);
		clReleaseDevice(oclData->intelDeviceID);
		clReleaseContext(oclData->context);
		clReleaseKernel(kernel);
		return -1;
	}
	
	
	
	/* Return. */
	return 0;
}


/********************* Global network code ***********************/

/**
 * Create server socket.
 */

int  createSocket(SSOCK_DATA* ssock){
	ssock->fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, IPPROTO_TCP);
	if(ssock->fd < 0){
		printf("Failed to create UDP socket!\n");
		return -1;
	}else{
		printf("Created UDP server socket.\n");
	}
	
	return 0;
}

/**
 * Close server socket.
 */

void closeSocket(SSOCK_DATA* ssock){
	close(ssock->fd);
	printf("Closed UDP server socket.\n");
}


/********************* Per-connection code ***********************/

/**
 * Accept connection.
 */

CONN_CTX* acceptConnection(SSOCK_DATA* ssock){
	/* Allocate connection context */
	CONN_CTX* connCtx = (CONN_CTX*)calloc(1, sizeof(*connCtx));
	
	
	return connCtx;
}

/**
 * Handle the connection's lifetime.
 */

void handleConnection(OCL_DATA* oclData, CONN_CTX* connCtx){
	/**
	 * OpenCL per-connection setup.
	 */
	
	printf("Args: clCCQ(%p, %p, %p, %p)\n",
	       oclData->context,
	       oclData->intelDeviceID,
	       (void*)NULL,
	       (void*)NULL);
	connCtx->cmdQ   = clCreateCommandQueue(oclData->context,
	                                       oclData->intelDeviceID,
	                                       0,
	                                       NULL);
	printf("Args: clCK(%p, %p)\n",
	       oclData->kernel,
	       (void*)NULL);
	connCtx->kernel = clCloneKernel(oclData->kernel, NULL);
	
	/* Allocate destination memory */
	float* dst    = (float*)malloc(192*192*128*sizeof(float));
	float* src    = (float*)malloc(256*256*128*sizeof(float));
	float* Harray = (float*)malloc(  3*  3*128*sizeof(float));
	
	printf("Args: clSKA(%p, %d, %zu, %p)\n",
	       oclData->kernel,
	       0,
	       sizeof(dst),
	       dst);
	clSetKernelArg(connCtx->kernel, 0, sizeof(dst), dst);
	printf("Args: clSKA(%p, %d, %zu, %p)\n",
	       oclData->kernel,
	       1,
	       sizeof(src),
	       src);
	clSetKernelArg(connCtx->kernel, 1, sizeof(src), src);
	printf("Args: clSKA(%p, %d, %zu, %p)\n",
	       oclData->kernel,
	       2,
	       sizeof(Harray),
	       Harray);
	clSetKernelArg(connCtx->kernel, 2, sizeof(Harray), Harray);
	
	/* Dimensions:               ( x , y , B ) */
	printf("Enqueuing...\n");
	size_t INPUT_TENSOR_SIZE[] = {192,192,128};
	clEnqueueNDRangeKernel(connCtx->cmdQ,
	                       connCtx->kernel,
	                       3,
	                       NULL,
	                       INPUT_TENSOR_SIZE,
	                       NULL,
	                       0,
	                       NULL,
	                       NULL);
	
	/**
	 * Tear-down.
	 */
	
	clReleaseCommandQueue(connCtx->cmdQ);
	clReleaseKernel(connCtx->kernel);
}


/**
 * Event loop.
 * 
 * Accept client, handle requests.
 */

void eventLoop(OCL_DATA* oclData){
	/* Create server socket */
	SSOCK_DATA ssock;
	if(createSocket(&ssock) < 0){
		return;
	}
	
	/* Accept clients */
	CONN_CTX* connCtx;
	while((connCtx = acceptConnection(&ssock))){
		handleConnection(oclData, connCtx);
	}
	
	/* Close socket. */
	closeSocket(&ssock);
}



/**
 * Main
 */

int main(int argc, char* argv[]){
	OCL_DATA oclData;
	
	(void)argc;
	(void)argv;
	
	/* Set up base OpenCL environment. */
	oclSetup(&oclData);
	
	/* Run event loop. */
	eventLoop(&oclData);
	
	return 0;
}

