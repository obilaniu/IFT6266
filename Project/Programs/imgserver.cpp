/**
 * Image Server.
 * 
 * 1. INTRODUCTION
 * 
 * Runs a server on TCP port 5555, accepting connections and serving over
 * shared memory images warped by the Intel GPU.
 * 
 * 2. STARTUP
 * 
 * At startup, the program takes two arguments:
 * 
 *     ./imgserver PathToHDF5File LocalTCPPort#
 * 
 * The first is the path to the catsanddogs.hdf5 file whose contents are to be
 * served. The second is the local port to bind to.
 * 
 * The program finds an Intel OpenCL implementation with an Intel iGPU, and
 * builds an OpenCL kernel.
 * 
 * The program sets up a TCP server socket on the LocalTCPPort#.
 * 
 * The program loads, then memory-maps the entire dataset using the HDF5 API.
 * 
 * 3. RUNTIME
 * 
 * The program waits on the server socket for connection attempts. When a
 * connection is made, it services the requests coming from the remote end and
 * acknowledges them.
 * 
 * 4. PROTOCOL
 * 4.1. START PACKET
 * 
 * Form: 0x00
 * Ack:  0x80
 * 
 * 4.2. CONFIG PACKET
 * 
 * Form: 0x01 [uint32_t cfgRqNum] {data...}
 * Ack:  0x81
 * 
 * 4.3. DATA REQUEST PACKET
 * 
 * Form: 0x10          (Training,   Random,     Warped,   Buffer 0)
 * Form: 0x11          (Training,   Random,     Warped,   Buffer 1)
 * Form: 0x12          (Training,   Random,     Unwarped, Buffer 0)
 * Form: 0x13          (Training,   Random,     Unwarped, Buffer 1)
 * Form: 0x14          (Training,   Non-Random, Warped,   Buffer 0)
 * Form: 0x15          (Training,   Non-Random, Warped,   Buffer 1)
 * Form: 0x16          (Training,   Non-Random, Unwarped, Buffer 0)
 * Form: 0x17          (Training,   Non-Random, Unwarped, Buffer 1)
 * Form: 0x18          (Validation, Random,     Warped,   Buffer 0)
 * Form: 0x19          (Validation, Random,     Warped,   Buffer 1)
 * Form: 0x1A          (Validation, Random,     Unwarped, Buffer 0)
 * Form: 0x1B          (Validation, Random,     Unwarped, Buffer 1)
 * Form: 0x1C          (Validation, Non-Random, Warped,   Buffer 0)
 * Form: 0x1D          (Validation, Non-Random, Warped,   Buffer 1)
 * Form: 0x1E          (Validation, Non-Random, Unwarped, Buffer 0)
 * Form: 0x1F          (Validation, Non-Random, Unwarped, Buffer 1)
 * Ack:  Above | 0x80
 * 
 * The buffer# indicates which buffer, out of two, to write fresh data in. For
 * instance, if the remote process is running a double buffer during training,
 * then before it processes buffer 0 it should make the request 0x11 for
 * training data to be written to buffer 1. When the remote process finishes
 * with buffer 0, it should wait for the Ack 0x91, make the request 0x10 for
 * new training data in buffer 0, then start processing buffer 1.
 * 
 * 4.4. EXIT PACKET
 * 
 * Form: 0x7F
 * Ack:  0xFF
 */

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
#include <sys/socket.h>
#include <arpa/inet.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


/* Defines */
#define OCL_INTEL_PLATFORM    "Intel"
#define OCL_INTEL_DEVICE      "Intel(R) HD Graphics Haswell GT2 Mobile"
#define DEFAULT_PORT          (5555)


/* Data structures */

/**
 * Global data.
 */

typedef struct{
	/* Program arguments */
	int                argc;
	const char**       argv;
	const char*        dataPath;
	
	/* OpenCL globals */
	cl_platform_id     clIntelPlatformID;
	cl_device_id       clIntelDeviceID;
	cl_context         clContext;
	cl_program         clProgram;
	
	/* Networking globals */
	uint16_t           localPort;
	sockaddr_in        localAddr;
	size_t             localAddrLen;
	int                ssockFd;
	
	/* Data globals */
	unsigned long long dataFileLen;
	int                dataFd;
	void*              dataPtr;
	float            (*dataY)[2];
	uint8_t          (*dataX_64x64)[64][64][3];
	uint8_t          (*dataX_128x128)[128][128][3];
	uint8_t          (*dataX_256x256)[256][256][3];
} GLOBAL_DATA;

/**
 * Per-connection context.
 */

typedef struct{
	/* Status locals */
	int              exit;
	
	/* Config */
	int              cfgParamsChanged;
	int              cfgImgSrcSize;
	int              cfgImgDstSize;
	int              cfgBatchSize;
	void*            cfgBufBounceIn;
	float          (*cfgBufH)[3][3];
	void*            cfgBufBounceOut;
	void*            cfgDevPtr;
	uint64_t         xorshift128p[2];
	
	/* OpenCL locals */
	cl_command_queue clCmdQ;
	cl_kernel        clWarpK;
	cl_kernel        clRgbaK;
	cl_image_format  clImgTmpFormat;
	cl_image_desc    clImgTmpDesc;
	cl_mem           clBufIn;
	cl_mem           clBufH;
	cl_mem           clImgTmp;
	cl_mem           clBufOut;
	
	/* Network locals */
	int              sockFd;
	sockaddr_in      remoteAddr;
	socklen_t        remoteAddrLen;
} CONN_CTX;

/**
 * Copies of the stdout and stderr file descriptors.
 */

int copyOut = -1, copyErr = -1, devnull = -1;


/* Functions */
/********************* Utilities ***********************/

/**
 * Muzzle stdin/stdout. Beignet's spam is extremely aggravating.
 */

void muzzle(void){
	if(copyOut == devnull){
		copyOut = dup(1);
		copyErr = dup(2);
		dup2(devnull, 1);
		dup2(devnull, 2);
	}
}

/**
 * Unmuzzle stdin/stdout. Beignet's spam is extremely aggravating.
 */

void unmuzzle(void){
	if(copyOut != devnull){
		dup2(copyOut, 1);
		dup2(copyErr, 2);
		close(copyOut);
		close(copyErr);
		copyOut = devnull;
		copyErr = devnull;
	}
}

/**
 * Initialize file descriptors.
 * 
 * Open /dev/null
 */

void setupMuzzle(void){
	devnull = open("/dev/null", O_WRONLY|O_CLOEXEC);
	copyErr = copyOut = devnull;
}

/**
 * Tear down file descriptors.
 */

void teardownMuzzle(void){
	if(copyOut != devnull){
		unmuzzle();
	}
	close(devnull);
	devnull = -1;
}

/**
 * PRNG.
 * 
 * xorshift128+
 */

double connRandom(CONN_CTX* connCtx, double mn, double mx){
	/* Load */
	uint64_t s0               = connCtx->xorshift128p[0];
	uint64_t s1               = connCtx->xorshift128p[1];
	/* Swap */
	uint64_t ns0              = s1;
	uint64_t ns1              = s0;
	/* Modify ns1 */
	ns1                      ^= ns1 << 23;
	ns1                      ^= ns1 >> 17;
	ns1                      ^= s1  >> 26;
	ns1                      ^= s1;
	/* Writeback */
	connCtx->xorshift128p[0]  = ns0;
	connCtx->xorshift128p[1]  = ns1;
	/* Distill */
	uint64_t v                = ns0 + ns1;
	/* Convert */
	double x = v / (65536.0*65536.0*65536.0*65536.0);
	/* Return */
	return mn+x*(mx-mn);
}

/**
 * Matrix mul 3x3
 * 
 * D = A*B
 */

void mmulH(float(*D)[3], float(*A)[3], float(*B)[3]){
#define DOT(i,j)                      \
	do{                               \
		C[i][j] = A[i][0]*B[0][j] +   \
		          A[i][1]*B[1][j] +   \
		          A[i][2]*B[2][j];    \
	}while(0)
#define COPYR(D,S,i)                  \
	do{                               \
		D[i][0] = S[i][0];            \
		D[i][1] = S[i][1];            \
		D[i][2] = S[i][2];            \
	}while(0)
#define COPY(D,S)     \
	do{               \
		COPYR(D,S,0); \
		COPYR(D,S,1); \
		COPYR(D,S,2); \
	}while(0)
	
	float C[3][3];
	
	DOT(0,0);
	DOT(0,1);
	DOT(0,2);
	
	DOT(1,0);
	DOT(1,1);
	DOT(1,2);
	
	DOT(2,0);
	DOT(2,1);
	DOT(2,2);
	
	COPY(D,C);
}

/**
 * Sample a random H within given limits.
 * 
 * - Translate up/down and left/right by no more than
 *   +- maxT.
 * - Rotate left/right by no more than maxR degrees.
 * - Scale by a factor in the range [minS, maxS].
 * - H must be adjusted for an input of sizeIn and an output of sizeOut.
 * 
 * The equation for sampling is the following:
 * 
 *     coord_src = H coord_dst
 * 
 * , where:
 * 
 *     H = Tran_{T}    *
 *         N_{src}     *
 *         Tran_{+0.5} *
 *         Rot_{R}     *
 *         Tran_{-0.5} *
 *         (1/Scal)    *
 *         (1/N_{dst})
 */

void sampleH(CONN_CTX* connCtx,
             float (*H)[3],
             float maxT,
             float maxR,
             float minS,
             float maxS,
             float sizeIn,
             float sizeOut){
	float Tx = connRandom(connCtx, -maxT, +maxT);
	float Ty = connRandom(connCtx, -maxT, +maxT);
	float R  = connRandom(connCtx, -maxR, +maxR) / (180.0/3.14159265358979323846483373);
	float S  = connRandom(connCtx,  minS,  maxS);
	float c  = cos(R);
	float s  = sin(R);
	
	float TranT[3][3] = {
	    1.0, 0.0, Tx,
	    0.0, 1.0, Ty,
	    0.0, 0.0, 1.0
	};
	float N_src[3][3] = {
	    sizeIn, 0.0,    0.0,
	    0.0,    sizeIn, 0.0,
	    0.0,    0.0,    1.0
	};
	float Tranp05[3][3] = {
	    1.0, 0.0, 0.5,
	    0.0, 1.0, 0.5,
	    0.0, 0.0, 1.0
	};
	float RotR[3][3] = {
	    c,   s,   0.0,
	    -s,  c,   0.0,
	    0.0, 0.0, 1.0
	};
	float Tranm05[3][3] = {
	    1.0, 0.0, -0.5,
	    0.0, 1.0, -0.5,
	    0.0, 0.0, 1.0
	};
	float InvScal[3][3] = {
	    1/S, 0.0, 0.0,
	    0.0, 1/S, 0.0,
	    0.0, 0.0, 1.0
	};
	float InvN_dst[3][3] = {
	    1/sizeOut, 0.0,       0.0,
	    0.0,       1/sizeOut, 0.0,
	    0.0,       0.0,       1.0
	};
	
	mmulH(H, TranT, N_src);
	mmulH(H, H,     Tranp05);
	mmulH(H, H,     RotR);
	mmulH(H, H,     Tranm05);
	mmulH(H, H,     InvScal);
	mmulH(H, H,     InvN_dst);
}


/********************* Global OpenCL state code ***********************/

/**
 * Load OpenCL source code.
 */

void loadOpenCLSourceCode(const char** SOURCECODE, size_t* SOURCECODELEN){
	*SOURCECODE = 
	"kernel void warpImage(__global float*           dst,\n"
	"                      read_only image2d_array_t src,\n"
	"                      __constant float*         Harray){\n"
	"    /* Get IDs. */\n"
	"    size_t x     = get_global_id(0);\n"
	"    size_t y     = get_global_id(1);\n"
	"    size_t b     = get_global_id(2);\n"
	"    size_t sX    = get_global_size(0);\n"
	"    size_t sY    = get_global_size(1);\n"
	"    size_t sB    = get_global_size(2);\n"
	"    \n"
	"    \n"
	"    /* Apply transformation matrix. */\n"
	"    __constant float(*H)[3] = (__constant float(*)[3])&Harray[b*3*3];\n"
	"    float3 row0  = (float3)(H[0][0],H[0][1],H[0][2]);\n"
	"    float3 row1  = (float3)(H[1][0],H[1][1],H[1][2]);\n"
	"    float3 row2  = (float3)(H[2][0],H[2][1],H[2][2]);\n"
	"    float3 v     = (float3)(x,y,1.0f);\n"
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
	"     */\n"
	"    \n"
	"    const sampler_t smpl = CLK_NORMALIZED_COORDS_FALSE |\n"
	"                           CLK_ADDRESS_CLAMP           |\n"
	"                           CLK_FILTER_LINEAR           ;\n"
	"    float3 rgb = read_imagef(src, smpl, coord).xyz;\n"
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
	"}\n"
	"\n"
	"\n"
	"\n"
	"kernel void bgr2rgba(write_only image2d_array_t dst,\n"
	"                     __global const uchar*      src){\n"
	"    /* Get IDs. */\n"
	"    size_t x     = get_global_id(0);\n"
	"    size_t y     = get_global_id(1);\n"
	"    size_t b     = get_global_id(2);\n"
	"    size_t sX    = get_global_size(0);\n"
	"    size_t sY    = get_global_size(1);\n"
	"    size_t sB    = get_global_size(2);\n"
	"    \n"
	"    \n"
	"    /* Read, BGR->RGB->RGBA, and write. */\n"
	"    __global const uchar* base = &src[x*3 + y*sX*3 + b*sX*sY*3];\n"
	"    uchar3 p3u = (uchar3)(base[0], base[1], base[2]);\n"
	"    float3 v   = convert_float3(p3u) * (1.0f/255.0f);\n"
	"    float4 p4f = (float4)(v, 1.0f);\n"
	"    write_imagef(dst, (int4)(x,y,b,0), p4f.zyxw);\n"
	"}\n"
	;
	*SOURCECODELEN = strlen(*SOURCECODE);
}

/**
 * OpenCL setup.
 */

int  setupOpenCL(GLOBAL_DATA* gData){
	/**
	 * NULL out everything.
	 */
	
	gData->clIntelPlatformID = NULL;
	gData->clIntelDeviceID = NULL;
	gData->clContext = NULL;
	gData->clProgram = NULL;
	
	/**
	 * Get # of OpenCL platforms.
	 */
	
	cl_uint NUM_PLATFORMS;
	muzzle();
	clGetPlatformIDs(0, NULL, &NUM_PLATFORMS);
	unmuzzle();
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
		muzzle();
		clGetPlatformInfo(platformIDs[i],
		                  CL_PLATFORM_VENDOR,
		                  sizeof(name),
		                  name,
		                  NULL);
		unmuzzle();
		
		if(strcmp(name, OCL_INTEL_PLATFORM) == 0){
			gData->clIntelPlatformID = platformIDs[i];
			break;
		}
	}
	
	if(gData->clIntelPlatformID){
		printf("Found Intel OpenCL platform.\n");
	}else{
		printf("ERROR: Failed to find Intel platform!\n");
		return -1;
	}
	
	
	/**
	 * Get # of Intel GPU devices.
	 */
	
	cl_uint NUM_DEVICES;
	muzzle();
	clGetDeviceIDs(gData->clIntelPlatformID,
	               CL_DEVICE_TYPE_GPU,
	               0,
	               NULL,
	               &NUM_DEVICES);
	unmuzzle();
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
	muzzle();
	clGetDeviceIDs(gData->clIntelPlatformID,
	               CL_DEVICE_TYPE_GPU,
	               NUM_DEVICES,
	               deviceIDs,
	               NULL);
	unmuzzle();
	
	
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
			gData->clIntelDeviceID = deviceIDs[i];
			break;
		}
	}
	
	if(gData->clIntelDeviceID){
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
	    (cl_context_properties)gData->clIntelPlatformID,
	    0
	};
	muzzle();
	gData->clContext = clCreateContext(clCtxProps,
	                                 1,
	                                 &gData->clIntelDeviceID,
	                                 NULL,
	                                 NULL,
	                                 NULL);
	unmuzzle();
	
	if(gData->clContext){
		printf("Created OpenCL context.\n");
	}else{
		printf("ERROR: Failed to create OpenCL context!\n");
		return -1;
	}
	
	
	
	/**
	 * Create OpenCL program out of source code.
	 */
	
	const char* SOURCECODE;
	size_t      SOURCECODELEN;
	loadOpenCLSourceCode(&SOURCECODE, &SOURCECODELEN);
	gData->clProgram = clCreateProgramWithSource(gData->clContext,
	                                           1,
	                                           &SOURCECODE,
	                                           &SOURCECODELEN,
	                                           NULL);
	
	if(gData->clProgram){
		printf("Created OpenCL program.\n");
	}else{
		printf("Error: Failed to create OpenCL program!\n");
		return -1;
	}
	
	
	
	/**
	 * Build OpenCL program.
	 */
	
	if(clBuildProgram(gData->clProgram,
	                  0,
	                  NULL,
	                  "-cl-std=CL1.2 -cl-denorms-are-zero -cl-mad-enable -cl-fast-relaxed-math",
	                  NULL,
	                  NULL) == CL_SUCCESS){
		printf("Built OpenCL program.\n");
	}else{
		printf("Error: Failed to build OpenCL program!\n");
		char buf[8192];
		clGetProgramBuildInfo(gData->clProgram,
		                      gData->clIntelDeviceID,
		                      CL_PROGRAM_BUILD_LOG,
		                      sizeof(buf),
		                      buf,
		                      0);
		printf("***********************************************************************\n");
		printf("%s\n", buf);
		printf("***********************************************************************\n");
		return -1;
	}
	clUnloadPlatformCompiler(gData->clIntelPlatformID);
	
	
	/**
	 * Check that the warpImage and bgr2rgba kernels are present in the source code.
	 */
	
	cl_kernel warpK = clCreateKernel(gData->clProgram, "warpImage", NULL);
	cl_kernel rgbaK = clCreateKernel(gData->clProgram, "bgr2rgba",  NULL);
	if(warpK && rgbaK){
		printf("Created \"warpImage\" and \"bgr2rgba\" kernels.\n");
		clReleaseKernel(warpK);
		clReleaseKernel(rgbaK);
	}else{
		printf("ERROR: Failed to create \"warpImage\" and \"bgr2rgba\" kernels!\n");
		clReleaseKernel(warpK);
		clReleaseKernel(rgbaK);
		return -1;
	}
	
	
	
	/**
	 * Return.
	 */
	
	return 0;
}

/**
 * OpenCL teardown.
 */

void teardownOpenCL(GLOBAL_DATA* gData){
	clReleaseProgram(gData->clProgram);
	clReleaseDevice(gData->clIntelDeviceID);
	clReleaseContext(gData->clContext);
}


/********************* Global network code ***********************/

/**
 * Ready server socket to accept connections.
 */

int  setupNetwork(GLOBAL_DATA* gData){
	/**
	 * Create socket.
	 */
	
	gData->ssockFd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, IPPROTO_TCP);
	if(gData->ssockFd < 0){
		printf("Failed to create TCP socket!\n");
		return -1;
	}else{
		printf("Created TCP server socket.\n");
	}
	
	
	/**
	 * Bind it.
	 */
	
	gData->localAddr.sin_family      = AF_INET;
	gData->localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	gData->localAddr.sin_port        = htons(gData->localPort);
	gData->localAddrLen              = sizeof(sockaddr_in);
	if(bind(gData->ssockFd,
	        (const sockaddr*)&gData->localAddr,
	        gData->localAddrLen) < 0){
		printf("Failed to bind TCP socket to port %d!\n", gData->localPort);
		close(gData->ssockFd);
		return -1;
	}else{
		printf("Bound TCP socket to port %d.\n", gData->localPort);
	}
	
	
	/**
	 * Listen on it.
	 */
	
	listen(gData->ssockFd, 1);
	
	return 0;
}

/**
 * Close server socket.
 */

void teardownNetwork(GLOBAL_DATA* gData){
	close(gData->ssockFd);
	printf("Closed TCP server socket.\n");
}


/********************** Global HDF5 code *************************/

/**
 * Load data.
 */

int  setupData(GLOBAL_DATA* gData){
	/**
	 * Open HDF5 file.
	 */
	
	hsize_t fSize;
	hid_t f = H5Fopen(gData->dataPath, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(f<0){
		gData->dataFileLen = 0;
		printf("Failed to open %s as HDF5 file!\n", gData->dataPath);
		return -1;
	}else{
		H5Fget_filesize(f, &fSize);
		gData->dataFileLen = fSize;
		printf("Opened %llu-bytes-long HDF5 file %s ...\n",
		       gData->dataFileLen, gData->dataPath);
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
	
	
	/**
	 * Open the file again, this time raw.
	 */
	
	gData->dataFd  = open(gData->dataPath, O_RDONLY|O_CLOEXEC);
	if(gData->dataFd < 0){
		printf("Failed to open file descriptor for %s!\n", gData->dataPath);
		return -1;
	}else{
		printf("Opened a file descriptor for %s.\n", gData->dataPath);
	}
	
	
	/**
	 * Memory-map the file.
	 */
	
	gData->dataPtr = mmap(NULL,
	                      fSize,
	                      PROT_READ,
	                      MAP_SHARED,
	                      gData->dataFd,
	                      0);
	if(gData->dataPtr == MAP_FAILED){
		printf("%5d: %s", errno, strerror(errno));
		printf("Failed to memory-map the file!\n");
		close(gData->dataFd);
		gData->dataFd = -1;
		return -1;
	}else{
		printf("Memory-mapped the file.\n");
	}
	
	/**
	 * Compute the pointers.
	 */
	
	char* base = (char*)gData->dataPtr;
	gData->dataY         = (float  (*)[  2]        )(base + offy);
	gData->dataX_64x64   = (uint8_t(*)[ 64][ 64][3])(base + offx_64x64);
	gData->dataX_128x128 = (uint8_t(*)[128][128][3])(base + offx_128x128);
	gData->dataX_256x256 = (uint8_t(*)[256][256][3])(base + offx_256x256);
	
	
	#if 0
	/**
	 * Temporary: Visualize. C++.
	 */
	
	for(int i=0;i<25000;i++){
		cv::Mat img = cv::Mat(64,64,CV_8UC3, &gData->dataX_64x64[i][0][0][0]);
		cv::imshow("Image", img);
		cv::waitKey(30);
	}
	#endif
	
	/* Return */
	return 0;
}

/**
 * Teardown data.
 */

void teardownData(GLOBAL_DATA* gData){
	munmap(gData->dataPtr, gData->dataFileLen);
	close(gData->dataFd);
}


/********************* Per-connection code ***********************/

/**
 * Accept connection.
 */

CONN_CTX* acceptConnection(GLOBAL_DATA* gData){
	/**
	 * Allocate connection context
	 */
	
	CONN_CTX* connCtx = (CONN_CTX*)calloc(1, sizeof(*connCtx));
	if(!connCtx){
		printf("Failed to allocate resources for connection!\n");
		return NULL;
	}
	
	/**
	 * Initialize PRNG
	 */
	
	connCtx->xorshift128p[0] =  0xDEADBEEF;
	connCtx->xorshift128p[1] = ~0xDEADBEEF;
	
	
	/**
	 * accept() a new connection.
	 */
	
	connCtx->remoteAddrLen = sizeof(connCtx->remoteAddr);
	connCtx->sockFd        = -1;
#if 0
	connCtx->sockFd        = accept(gData->ssockFd,
	                                (sockaddr*)&connCtx->remoteAddr,
	                                &connCtx->remoteAddrLen);
	if(connCtx->sockFd < 0){
		printf("Server socket failed!\n");
		free(connCtx);
		return NULL;
	}else{
		unsigned char* addr = (unsigned char*)&connCtx->remoteAddr.sin_addr.s_addr;
		unsigned       port = htons(connCtx->remoteAddr.sin_port);
		printf("Received new connection from %u.%u.%u.%u:%u\n",
		       addr[0], addr[1], addr[2], addr[3], port);
	}
#endif
	
	return connCtx;
}

/**
 * (Maybe) reallocate OpenCL buffers.
 */

int  maybeReallocateBufs(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	int err;
	
	/**
	 * If the settings are invalid, don't reallocate.
	 */
	
	if(connCtx->cfgBatchSize <= 0         ||
	   (connCtx->cfgImgSrcSize != 64  &&
	    connCtx->cfgImgSrcSize != 128 &&
	    connCtx->cfgImgSrcSize != 256   ) ||
	    connCtx->cfgImgSrcSize <= 0){
		return -2;
	}
	
	
	/**
	 * If the settings haven't changed, don't reallocate.
	 */
	
	if(!connCtx->cfgParamsChanged){
		return 0;
	}
	
	
	/**
	 * Otherwise, reallocate everything.
	 */
	
	const int B  = connCtx->cfgBatchSize,
	          Wi = connCtx->cfgImgSrcSize,
	          Hi = connCtx->cfgImgSrcSize,
	          Wo = connCtx->cfgImgDstSize,
	          Ho = connCtx->cfgImgDstSize;
	
	free(connCtx->cfgBufBounceIn);
	free(connCtx->cfgBufH);
	free(connCtx->cfgBufBounceOut);
	clReleaseMemObject(connCtx->clBufIn);
	clReleaseMemObject(connCtx->clBufH);
	clReleaseMemObject(connCtx->clBufOut);
	clReleaseMemObject(connCtx->clImgTmp);
	connCtx->cfgBufBounceIn  = calloc(B*Wi*Hi*3, sizeof(uint8_t));
	connCtx->cfgBufH         = (float(*)[3][3])calloc(B* 3* 3, sizeof(float));
	connCtx->cfgBufBounceOut = calloc(B*Wo*Ho*3, sizeof(float));
	
	/**
	 * OpenCL per-connection setup.
	 * 
	 * Each connection gets a command queue and its own instance of the
	 * kernels. The data flow is:
	 * 
	 * HOST:   Image selection --- Warp Selection     CUDA pinned memory
	 *                \                    \             /         \
	 *                 \                    \           /           \
	 *                  \                    \         /             \
	 * iGPU:             bgr2rgba --imgTmp--- warpImage               \
	 *                                                                 \
	 *                                                                  \
	 *                                                                   \
	 * dGPU:                                                          Theano SV
	 * 
	 * On input the images are interleaved BGR uint8_t.
	 * 
	 * While on the device, the image takes the shape of an array of 2D images
	 * labelled in the graph above "imgTmp". It has the following
	 * specifications:
	 * 
	 *     - WxHxB
	 *     - RGBA
	 *     - CL_FLOAT
	 * 
	 * On output the images are planar RGB float32.
	 */
	
	
	connCtx->clImgTmpFormat.image_channel_order     = CL_RGBA;
	connCtx->clImgTmpFormat.image_channel_data_type = CL_FLOAT;
	
	connCtx->clImgTmpDesc.image_type                = CL_MEM_OBJECT_IMAGE2D_ARRAY;
	connCtx->clImgTmpDesc.image_width               = Wi; /* Image width */
	connCtx->clImgTmpDesc.image_height              = Hi; /* (2D) Image height */
	connCtx->clImgTmpDesc.image_depth               = 1;  /* (3D) Image depth */
	connCtx->clImgTmpDesc.image_array_size          = B;  /* (Array) Size */
	connCtx->clImgTmpDesc.image_row_pitch           = 0;
	connCtx->clImgTmpDesc.image_slice_pitch         = 0;
	connCtx->clImgTmpDesc.num_mip_levels            = 0;
	connCtx->clImgTmpDesc.num_samples               = 0;
	connCtx->clImgTmpDesc.buffer                    = NULL;
	
	connCtx->clBufIn  = clCreateBuffer(gData->clContext,
	                                   CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, 
	                                   B*Wi*Hi*3*sizeof(uint8_t),
	                                   connCtx->cfgBufBounceIn,
	                                   NULL);
	connCtx->clBufH   = clCreateBuffer(gData->clContext,
	                                   CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, 
	                                   B* 3* 3 * sizeof(float),
	                                   connCtx->cfgBufH,
	                                   NULL);
	connCtx->clBufOut = clCreateBuffer(gData->clContext,
	                                   CL_MEM_WRITE_ONLY |
	                                   CL_MEM_USE_HOST_PTR,
	                                   B*Wo*Ho*3*sizeof(float),
	                                   connCtx->cfgBufBounceOut,
	                                   NULL);
	connCtx->clImgTmp = clCreateImage(gData->clContext,
	                                  CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
	                                  &connCtx->clImgTmpFormat,
	                                  &connCtx->clImgTmpDesc,
	                                  NULL,
	                                  &err);
	if(!connCtx->clBufIn || !connCtx->clBufH || !connCtx->clBufOut){
		printf("Failed to create input and output buffers!\n");
		return -1;
	}
	if(err != CL_SUCCESS){
		printf("Failed to create temporary image!\n");
		if(err == CL_INVALID_CONTEXT){
			printf("\t-> Invalid Context.\n");
		}else if(err == CL_INVALID_VALUE){
			printf("\t-> Invalid Value.\n");
		}else if(err == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR){
			printf("\t-> Invalid Image Format Descriptor.\n");
		}else if(err == CL_INVALID_IMAGE_DESCRIPTOR){
			printf("\t-> Invalid Image Descriptor.\n");
		}else if(err == CL_INVALID_IMAGE_SIZE){
			printf("\t-> Invalid Image Size.\n");
		}else if(err == CL_INVALID_HOST_PTR){
			printf("\t-> Invalid Host Pointer.\n");
		}
		
		return -1;
	}
	
	return 0;
}

/**
 * Setup connection.
 */

int  setupConnection(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	/* Status setup */
	connCtx->exit             = 0;
	
	
	/* OpenCL setup */
	connCtx->clCmdQ   = clCreateCommandQueue(gData->clContext,
	                                       gData->clIntelDeviceID,
	                                       0,
	                                       NULL);
	connCtx->clWarpK  = clCreateKernel(gData->clProgram, "warpImage", NULL);
	connCtx->clRgbaK  = clCreateKernel(gData->clProgram, "bgr2rgba",  NULL);
	if(!connCtx->clCmdQ || !connCtx->clWarpK || !connCtx->clRgbaK){
		printf("Failed to create OpenCL command queue or kernel!\n");
		return -1;
	}
	
	
	/* Config setup */
	connCtx->cfgParamsChanged = 0;
	connCtx->cfgImgSrcSize    = 0;
	connCtx->cfgImgDstSize    = 0;
	connCtx->cfgBatchSize     = 0;
	connCtx->cfgBufBounceIn   = NULL;
	connCtx->cfgBufH          = NULL;
	connCtx->cfgBufBounceOut  = NULL;
	connCtx->cfgDevPtr        = NULL;
	
	
	return 0;
}

/**
 * Handle one packet coming from the remote end of the connection.
 */

void handleConnectionPacket(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	/* TEST */
	/* Configure */
	connCtx->cfgBatchSize     = 1024;
	connCtx->cfgImgSrcSize    = 256;
	connCtx->cfgImgDstSize    = 256;
	connCtx->cfgParamsChanged =   1;
	
	const int B  = connCtx->cfgBatchSize,
	          Wi = connCtx->cfgImgSrcSize,
	          Hi = connCtx->cfgImgSrcSize,
	          Wo = connCtx->cfgImgDstSize,
	          Ho = connCtx->cfgImgDstSize;
	
	
	/* Reallocate */
	printf("Reallocating...\n");
	maybeReallocateBufs(gData, connCtx);
	
	/* Identity H */
	for(int i=0;i<B;i++){
		sampleH(connCtx, connCtx->cfgBufH[i],
		        16, 60, 0.75, 1.25, Wi, Wo);
		/*
		float (*H)[3] = connCtx->cfgBufH[i];
		printf("%f   %f   %f\n", H[0][0], H[0][1], H[0][2]);
		printf("%f   %f   %f\n", H[1][0], H[1][1], H[1][2]);
		printf("%f   %f   %f\n", H[2][0], H[2][1], H[2][2]);
		*/
	}
	
	
	printf("Running kernel...\n");
	for(int idx=0;idx<10;idx++){
	struct timespec ts, te;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	
	
	/* Upload data */
	const size_t INPUT_TENSOR_SIZE[]  = {(size_t)Wi,(size_t)Hi,(size_t)B};
	const size_t OUTPUT_TENSOR_SIZE[] = {(size_t)Wo,(size_t)Ho,(size_t)B};
	const size_t WARP_SIZE[]          = {4,5,1};
	
	
	memcpy(connCtx->cfgBufBounceIn, gData->dataX_256x256, B*Wi*Hi*3*sizeof(uint8_t));
	clEnqueueWriteBuffer(connCtx->clCmdQ,
	                     connCtx->clBufIn,
	                     1,
	                     0,
	                     B*Wi*Hi*3*sizeof(uint8_t),
	                     connCtx->cfgBufBounceIn,
	                     0,
	                     NULL,
	                     NULL);
	clEnqueueWriteBuffer(connCtx->clCmdQ,
	                     connCtx->clBufH,
	                     1,
	                     0,
	                     B*3*3*sizeof(float),
	                     connCtx->cfgBufH,
	                     0,
	                     NULL,
	                     NULL);
	
	/* Invoke bgr2rgba */
	clSetKernelArg(connCtx->clRgbaK, 0, sizeof(connCtx->clImgTmp), &connCtx->clImgTmp);
	clSetKernelArg(connCtx->clRgbaK, 1, sizeof(connCtx->clBufIn),  &connCtx->clBufIn);
	
	clEnqueueNDRangeKernel(connCtx->clCmdQ,
	                       connCtx->clRgbaK,
	                       3,
	                       NULL,
	                       INPUT_TENSOR_SIZE,
	                       NULL,
	                       0,
	                       NULL,
	                       NULL);
	
	/* Invoke warpImage */
	clSetKernelArg(connCtx->clWarpK, 0, sizeof(connCtx->clBufOut), &connCtx->clBufOut);
	clSetKernelArg(connCtx->clWarpK, 1, sizeof(connCtx->clImgTmp), &connCtx->clImgTmp);
	clSetKernelArg(connCtx->clWarpK, 2, sizeof(connCtx->clBufH),   &connCtx->clBufH);
	
	clEnqueueNDRangeKernel(connCtx->clCmdQ,
	                       connCtx->clWarpK,
	                       3,
	                       NULL,
	                       OUTPUT_TENSOR_SIZE,
	                       WARP_SIZE,
	                       0,
	                       NULL,
	                       NULL);
	
	/* Download data */
	clEnqueueReadBuffer(connCtx->clCmdQ,
	                    connCtx->clBufOut,
	                    1,
	                    0,
	                    B*Wo*Ho*3*sizeof(float),
	                    connCtx->cfgBufBounceOut,
	                    0,
	                    NULL,
	                    NULL);
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &te);
	double tsN = 1e9*ts.tv_sec + ts.tv_nsec;
	double teN = 1e9*te.tv_sec + te.tv_nsec;
	
	printf("Ran kernel. (%f)\n", (teN-tsN)/1e9);
	}
	
#if 1
	/**
	 * Temporary: Visualize. C++.
	 */
	
	char* base = (char*)connCtx->cfgBufBounceOut;
	for(int i=0;i<B;i++){
		using namespace cv;
		
		Mat img = Mat(Ho,Wo,CV_32FC1, base + i*3*Ho*Wo*sizeof(float));
		imshow("Image", img);
		if(waitKey(0) == 033){break;}
	}
#endif
	
	
	
	/**
	 * Read first byte of packet. If the client died or quit on us, pretend
	 * we received packet 0x7F (EXIT).
	 */
	
	char c;
	ssize_t numBytesRead;
	
	numBytesRead = read(connCtx->sockFd, &c, sizeof(c));
	if(numBytesRead <= 0){
		printf("Client abruptly disconnected. Pretending we've received "
		       "EXIT packet.\n");
		c = 0x7F;
	}
	
	/* Parse switch */
	switch(c){
		case 0x00:
			
		break;
		case 0x01:
			
		break;
		case 0x10:
		case 0x11:
		case 0x12:
		case 0x13:
		case 0x14:
		case 0x15:
		case 0x16:
		case 0x17:
		case 0x18:
		case 0x19:
		case 0x1A:
		case 0x1B:
		case 0x1C:
		case 0x1D:
		case 0x1E:
		case 0x1F:
			
		break;
		case 0x7F:
		default:
			connCtx->exit = 1;
		break;
	}
}

/**
 * Teardown connection.
 */

void teardownConnection(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	clReleaseCommandQueue(connCtx->clCmdQ);
	clReleaseKernel(connCtx->clWarpK);
	clReleaseKernel(connCtx->clRgbaK);
	clReleaseMemObject(connCtx->clBufIn);
	clReleaseMemObject(connCtx->clImgTmp);
	clReleaseMemObject(connCtx->clBufOut);
	
	close(connCtx->sockFd);
}

/**
 * Handle the connection's lifetime.
 */

void handleConnection(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	if(setupConnection(gData, connCtx) < 0){
		printf("Could not set up resources for new connection!\n");
		connCtx->exit = 1;
	}
	
	while(!connCtx->exit){
		handleConnectionPacket(gData, connCtx);
	}
	
	teardownConnection(gData, connCtx);
}

/**
 * Check and parse argument sanity
 */

int  checkandParseArgs(GLOBAL_DATA* gData){
	if(gData->argc != 2 && gData->argc != 3){
		printf("Usage: imgserver <path/to/catsanddogs.hdf5> {TCPPort#}\n");
		return -1;
	}
	
	struct stat fileStat;
	if(stat(gData->argv[1], &fileStat) < 0){
		printf("'%s' cannot be accessed!\n", gData->argv[1]);
		return -1;
	}
	
	if(!S_ISREG(fileStat.st_mode)){
		printf("'%s' is not a regular file!\n", gData->argv[1]);
		return -1;
	}else{
		gData->dataPath = gData->argv[1];
	}
	
	if(gData->argc == 2){
		gData->localPort = DEFAULT_PORT;
	}else{
		gData->localPort = strtoul(gData->argv[2], NULL, 0);
	}
	
	return 0;
}

/**
 * Global setup.
 */

int  globalSetup(GLOBAL_DATA* gData){
	/* Argument sanity check */
	if(checkandParseArgs(gData) < 0){
		printf("Arguments are insane!\n");
		return -1;
	}
	
	/* Set up muzzle for Beignet */
	setupMuzzle();
	
	/* Set up OpenCL environment. */
	if(setupOpenCL(gData) < 0){
		printf("Failure in OpenCL setup!\n");
		return -1;
	}
	
	/* Create server socket */
	if(setupNetwork(gData) < 0){
		printf("Failure in network setup!\n");
		return -1;
	}
	
	/* Load data */
	if(setupData(gData) < 0){
		printf("Failure in data load!\n");
		return -1;
	}
	
	return 0;
}

/**
 * Global teardown.
 */

void globalTeardown(GLOBAL_DATA* gData){
	teardownMuzzle();
	teardownOpenCL(gData);
	teardownNetwork(gData);
	teardownData(gData);
}


/**
 * Event loop.
 * 
 * Accept client, handle requests.
 */

int eventLoop(GLOBAL_DATA* gData){
	CONN_CTX* connCtx;
	
	while((connCtx = acceptConnection(gData))){
		handleConnection(gData, connCtx);
		break;
	}
	
	return 0;
}


/**
 * Main
 */

int main(int argc, char* argv[]){
	int         ret;
	GLOBAL_DATA gData;
	gData.argc = argc;
	gData.argv = (const char**)argv;
	
	/* Global setup */
	ret = globalSetup(&gData);
	if(ret < 0){
		printf("Failure in setup!\n");
		return ret;
	}
	
	/* Run server till it dies. */
	ret = eventLoop(&gData);
	
	/* Tear down. */
	globalTeardown(&gData);
	
	/* Return */
	return ret;
}

