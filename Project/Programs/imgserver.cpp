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

/* Do we want display? */
#define OCVDISP               1

/* Includes */
#include <hdf5.h>
#include <signal.h>
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
#if OCVDISP
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <npp.h>



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
	
	/* Status */
	int                exiting;
	
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
	uint8_t          (*dataX_64x64)  [3][ 64][ 64];
	uint8_t          (*dataX_128x128)[3][128][128];
	uint8_t          (*dataX_256x256)[3][256][256];
	
	/* CUDA */
	float            (*devDataY)[2];
	Npp8u            (*devDataX_64x64)  [3][ 64][ 64];
	Npp8u            (*devDataX_128x128)[3][128][128];
	Npp8u            (*devDataX_256x256)[3][256][256];
} GLOBAL_DATA;

/**
 * Per-connection context.
 */

typedef struct{
	/* Status locals */
	int              exit;
	
	/* Config */
	uint64_t         xorshift128p[2];
	
	/* CUDA */
	void*            devBounceBuf;
	
	/* Network locals */
	int              sockFd;
	sockaddr_in      remoteAddr;
	socklen_t        remoteAddrLen;
} CONN_CTX;

/**
 * Data Request Packet.
 */

typedef struct{
	uint64_t           req;          /* 0x00 */
	uint64_t           batchSize;    /* 0x08 */
	uint64_t           first;        /* 0x10 */
	uint64_t           last;         /* 0x18 */
	cudaIpcMemHandle_t Y;            /* 0x20 */
	cudaIpcMemHandle_t X;            /* 0x60 */
	uint64_t           sizeIn;       /* 0xA0 */
	uint64_t           sizeOut;      /* 0xA8 */
	uint64_t           x128ps0;      /* 0xB0 */
	uint64_t           x128ps1;      /* 0xB8 */
	double             maxT;         /* 0xC0 */
	double             maxR;         /* 0xC8 */
	double             minS;         /* 0xD0 */
	double             maxS;         /* 0xD8 */
} CONN_PKT;



/* Global Variables */
GLOBAL_DATA gData;



/* Forward Declarations */
int       imgsNetworkSetup(GLOBAL_DATA* gData);
void      imgsNetworkTeardown(GLOBAL_DATA* gData);
int       imgsDataSetup(GLOBAL_DATA* gData);
void      imgsDataTeardown(GLOBAL_DATA* gData);
int       imgsCUDASetup(GLOBAL_DATA* gData);
void      imgsCUDATeardown(GLOBAL_DATA* gData);
CONN_CTX* imgsConnAlloc(GLOBAL_DATA* gData);
CONN_CTX* imgsConnAccept(GLOBAL_DATA* gData);
int       imgsConnSetup(GLOBAL_DATA* gData, CONN_CTX* connCtx);
void      imgsConnHandlePacket(GLOBAL_DATA* gData, CONN_CTX* connCtx);
int       imgsConnReadPacket(GLOBAL_DATA* gData, CONN_CTX* connCtx, CONN_PKT* pkt);
uint64_t  imgsConnSelectImage(GLOBAL_DATA* gData,
                              CONN_CTX*    connCtx,
                              CONN_PKT*    pkt,
                              int          i);
void      imgsConnWantExit(CONN_CTX* connCtx);
int       imgsConnIsExitWanted(CONN_CTX* connCtx);
void      imgsConnTeardown(GLOBAL_DATA* gData, CONN_CTX* connCtx);
void      imgsConnHandle(GLOBAL_DATA* gData, CONN_CTX* connCtx);
int       imgsParseAndCheckArgs(GLOBAL_DATA* gData);
int       imgsGlobalSetup(GLOBAL_DATA* gData);
void      imgsGlobalTeardown(GLOBAL_DATA* gData);
void      sigHandler(int sig);
void      atexitHandler(void);



/* Functions */
/********************* Utilities ***********************/

/**
 * PRNG.
 * 
 * xorshift128+
 */

double imgsConnRandom(CONN_CTX* connCtx, double mn, double mx){
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

void mmulH(double (*D)[3], double (*A)[3], double (*B)[3]){
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
	
	double C[3][3];
	
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
 *     coord_dst = H coord_src
 * 
 * , where:
 * 
 *     H = Tran_{T}    *
 *         N_{src}     *
 *         Tran_{+0.5} *
 *         Rot_{R}     *
 *         (1/Scal)    *
 *         Tran_{-0.5} *
 *         (1/N_{dst})
 */

void imgsConnSampleH(CONN_CTX* connCtx,
                     double    (*H)[3],
                     double    maxT,
                     double    maxR,
                     double    minS,
                     double    maxS,
                     double    sizeIn,
                     double    sizeOut){
	double Tx = imgsConnRandom(connCtx, -maxT, +maxT);
	double Ty = imgsConnRandom(connCtx, -maxT, +maxT);
	double R  = imgsConnRandom(connCtx, -maxR, +maxR) / (180.0/3.14159265358979323846483373);
	double S  = imgsConnRandom(connCtx,  minS,  maxS);
	double Fl = imgsConnRandom(connCtx,     0,     1) < 0.5 ? -1.0 : 1.0;
	double c  = cos(R);
	double s  = sin(R);
	
	double TranT[3][3] = {
	    1.0, 0.0, Tx,
	    0.0, 1.0, Ty,
	    0.0, 0.0, 1.0
	};
	double N_src[3][3] = {
	    1/sizeIn, 0.0,      0.0,
	    0.0,      1/sizeIn, 0.0,
	    0.0,      0.0,      1.0
	};
	double Tranm05[3][3] = {
	    1.0, 0.0, -0.5,
	    0.0, 1.0, -0.5,
	    0.0, 0.0, +1.0
	};
	double RotR[3][3] = {
	    c,   s,   0.0,
	    -s,  c,   0.0,
	    0.0, 0.0, 1.0
	};
	double Tranp05[3][3] = {
	    1.0, 0.0, +0.5,
	    0.0, 1.0, +0.5,
	    0.0, 0.0, +1.0
	};
	double InvScal[3][3] = {
	 Fl*S,   0.0, 0.0,
	    0.0, S,   0.0,
	    0.0, 0.0, 1.0
	};
	double InvN_dst[3][3] = {
	    sizeOut,   0.0,       0.0,
	    0.0,       sizeOut,   0.0,
	    0.0,       0.0,       1.0
	};
	
	mmulH(H, InvN_dst, Tranp05);
	mmulH(H, H,        InvScal);
	mmulH(H, H,        RotR);
	mmulH(H, H,        Tranm05);
	mmulH(H, H,        N_src);
	mmulH(H, H,        TranT);
}


/********************* Global network code ***********************/

/**
 * Ready server socket to accept connections.
 */

int  imgsNetworkSetup(GLOBAL_DATA* gData){
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

void imgsNetworkTeardown(GLOBAL_DATA* gData){
	close(gData->ssockFd);
	printf("Closed TCP server socket.\n");
}


/********************** Global HDF5 code *************************/

/**
 * Load data.
 */

int  imgsDataSetup(GLOBAL_DATA* gData){
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
	
	gData->dataFd  = open(gData->dataPath, O_RDWR|O_CLOEXEC);
	if(gData->dataFd < 0){
		printf("Failed to open file descriptor for %s!\n", gData->dataPath);
		return -1;
	}
	
	
	/**
	 * Memory-map the file.
	 */
	
	gData->dataPtr = mmap(NULL,
	                      fSize,
	                      PROT_READ|PROT_WRITE,
	                      MAP_PRIVATE,
	                      gData->dataFd,
	                      0);
	if(gData->dataPtr == MAP_FAILED){
		int err = errno;
		printf("Failed to memory-map the file!\n");
		printf("->\t%5d: %s\n", err, strerror(err));
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
	gData->dataY         = (float  (*)[2]          )(base + offy);
	gData->dataX_64x64   = (uint8_t(*)[3][ 64][ 64])(base + offx_64x64);
	gData->dataX_128x128 = (uint8_t(*)[3][128][128])(base + offx_128x128);
	gData->dataX_256x256 = (uint8_t(*)[3][256][256])(base + offx_256x256);
	
	/* Return */
	return 0;
}

/**
 * Teardown data.
 */

void imgsDataTeardown(GLOBAL_DATA* gData){
	munmap(gData->dataPtr, gData->dataFileLen);
	close(gData->dataFd);
}


/********************** Global CUDA code *************************/

/**
 * Setup CUDA
 */

int  imgsCUDASetup(GLOBAL_DATA* gData){
	int devNum = 0;
	
	/* Select the device */
	if(cudaSetDevice(devNum) != cudaSuccess){
		printf("Could not select NVIDIA device %d!\n", devNum);
		return -1;
	}else{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, devNum);
		printf("Selected NVIDIA device %d (%s).\n", devNum, devProp.name);
		
	}
	
	/* Page-lock and memory-map the HDF5 file into the GPU's address space. */
	if(cudaHostRegister(gData->dataPtr,
	                    gData->dataFileLen,
	                    cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess){
		printf("Could not map dataset into GPU memory!\n->\t%s\n", cudaGetErrorString(cudaGetLastError()));
		return -1;
	}
	
	/* Get the device's pointers for the datasets. */
	if(cudaHostGetDevicePointer((void**)&gData->devDataY,         (void*)gData->dataY,         0) != cudaSuccess ||
	   cudaHostGetDevicePointer((void**)&gData->devDataX_64x64,   (void*)gData->dataX_64x64,   0) != cudaSuccess ||
	   cudaHostGetDevicePointer((void**)&gData->devDataX_128x128, (void*)gData->dataX_128x128, 0) != cudaSuccess ||
	   cudaHostGetDevicePointer((void**)&gData->devDataX_256x256, (void*)gData->dataX_256x256, 0) != cudaSuccess){
		printf("Error getting device pointers!\n");
		return -1;
	}
	
	return 0;
}

/**
 * Tear down CUDA
 */

void imgsCUDATeardown(GLOBAL_DATA* gData){
	cudaDeviceSynchronize();
	cudaHostUnregister(gData->dataPtr);
	cudaDeviceReset();
}


/********************* Per-connection code ***********************/

/**
 * Setup connection.
 */

CONN_CTX* imgsConnAlloc(GLOBAL_DATA* gData){
	/**
	 * Allocate connection context
	 */
	
	CONN_CTX* connCtx = (CONN_CTX*)calloc(1, sizeof(*connCtx));
	if(!connCtx){
		return NULL;
	}
	
	/* Status setup */
	connCtx->exit             = 0;
	
	/* PRNG */
	connCtx->xorshift128p[0]  =  0xDEADBEEF;
	connCtx->xorshift128p[1]  = ~0xDEADBEEF;
	
	/* Networking */
	connCtx->remoteAddrLen    = sizeof(connCtx->remoteAddr);
	connCtx->sockFd           = -1;
	
	/* Return */
	return connCtx;
}

/**
 * Accept connection.
 */

CONN_CTX* imgsConnAccept(GLOBAL_DATA* gData){
	/**
	 * Allocate connection context
	 */
	
	CONN_CTX* connCtx = imgsConnAlloc(gData);
	if(!connCtx){
		printf("Failed to allocate resources for connection!\n");
		return NULL;
	}
	
	/**
	 * Accept a new connection.
	 */
#if 1
	connCtx->sockFd        = accept(gData->ssockFd,
	                                (sockaddr*)&connCtx->remoteAddr,
	                                &connCtx->remoteAddrLen);
	if(connCtx->sockFd < 0){
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
 * Setup connection.
 */

int  imgsConnSetup(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	/* Allocate bounce buffer */
	if(cudaMalloc((void**)&connCtx->devBounceBuf, 3*256*256) != cudaSuccess){
		printf("Error allocating device bounce buffer!\n");
		return -1;
	}
	
	
	return 0;
}

/**
 * Handle one packet coming from the remote end of the connection.
 */

void imgsConnHandlePacket(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	/**
	__host__ ​cudaError_t cudaIpcCloseMemHandle ( void* devPtr )
	    Close memory mapped with cudaIpcOpenMemHandle. 
	__host__ ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags )
	    Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process
	 */
	
	
	/* Read the packet. */
	CONN_PKT pkt;
	if(imgsConnReadPacket(gData, connCtx, &pkt) <= 0){
		imgsConnWantExit(connCtx);return;
	}
	
	/* Sanity-check packet. */
	if(pkt.batchSize ==     0  ||
	   pkt.first     <      0  ||
	   pkt.first     >= 25000  ||
	   pkt.last      <      0  ||
	   pkt.last      >= 25000  ||
	   (pkt.sizeIn   !=  64    &&
	    pkt.sizeIn   != 128    &&
	    pkt.sizeIn   != 256)){
		printf("Insane packet!\n");
		printf("first   = %lu\n", pkt.first);
		printf("last    = %lu\n", pkt.last);
		printf("sizeIn  = %lu\n", pkt.sizeIn);
		printf("sizeOut = %lu\n", pkt.sizeOut);
		imgsConnWantExit(connCtx);return;
	}
	
	/* Attempt to obtain pointers for X and Y */
	void* devX = NULL, *devY = NULL;
	if(cudaIpcOpenMemHandle(&devX, pkt.X, cudaIpcMemLazyEnablePeerAccess) != cudaSuccess ||
	   cudaIpcOpenMemHandle(&devY, pkt.Y, cudaIpcMemLazyEnablePeerAccess) != cudaSuccess){
		cudaIpcCloseMemHandle(devX);
		cudaIpcCloseMemHandle(devY);
		printf("Failed to open IPC memory handles.\n");
		imgsConnWantExit(connCtx);return;
	}
	
	/* Wipe target buffer X */
	cudaMemsetAsync(devX, 0, pkt.batchSize*3*pkt.sizeOut*pkt.sizeOut, cudaStreamDefault);
	
	/* Precomputed work before warping starts... */
	NppiSize srcSize          = {      (int)pkt.sizeIn,  (int)pkt.sizeIn};
	NppiRect srcROI           = {0, 0, (int)pkt.sizeIn,  (int)pkt.sizeIn};
	NppiRect dstROI           = {0, 0, (int)pkt.sizeOut, (int)pkt.sizeOut};
	const Npp8u* srcPlanes[3] = {(const Npp8u*)connCtx->devBounceBuf + 0*pkt.sizeIn*pkt.sizeIn,
	                             (const Npp8u*)connCtx->devBounceBuf + 1*pkt.sizeIn*pkt.sizeIn,
	                             (const Npp8u*)connCtx->devBounceBuf + 2*pkt.sizeIn*pkt.sizeIn};
	Npp8u*       dstPlanes[3] = {NULL, NULL, NULL};
	double H[3][3]            = {{1,0,0},{0,1,0},{0,0,1}};
	int i;
	
	/**
	 * Loop over images, loading them into the bounce buffer and warping
	 * them from there to their destination.
	 */
	
	for(i=0;i<pkt.batchSize;i++){
		/* Select image according to some policy. */
		uint64_t n = imgsConnSelectImage(gData, connCtx, &pkt, i);/* Select some image somehow. */
		
		/* Copy selected image to GPU. */
		if(pkt.sizeIn == 64){
			cudaMemcpyAsync(connCtx->devBounceBuf, (void*)&gData->devDataX_64x64[n],
			                3* 64* 64, cudaMemcpyHostToDevice, cudaStreamDefault);
		}else if(pkt.sizeIn == 128){
			cudaMemcpyAsync(connCtx->devBounceBuf, (void*)&gData->devDataX_128x128[n],
			                3*128*128, cudaMemcpyHostToDevice, cudaStreamDefault);
		}else if(pkt.sizeIn == 256){
			cudaMemcpyAsync(connCtx->devBounceBuf, (void*)&gData->devDataX_256x256[n],
			                3*256*256, cudaMemcpyHostToDevice, cudaStreamDefault);
		}
		
		/* Memcpy label information into devY */
		cudaMemcpyAsync((void*)(&((float(*)[2])devY)[i]), (void*)&gData->devDataY[n],
		                2*sizeof(float), cudaMemcpyHostToDevice, cudaStreamDefault);
		
		/* Sample a homography from the selected parameters. */
		imgsConnSampleH(connCtx, H, pkt.maxT, pkt.maxR, pkt.minS, pkt.maxS,
		                pkt.sizeIn, pkt.sizeOut);
		
		/* Compute destination plane pointers into devX. */
		dstPlanes[0] = (Npp8u*)devX + (3*i+0)*pkt.sizeOut*pkt.sizeOut;
		dstPlanes[1] = (Npp8u*)devX + (3*i+1)*pkt.sizeOut*pkt.sizeOut;
		dstPlanes[2] = (Npp8u*)devX + (3*i+2)*pkt.sizeOut*pkt.sizeOut;
		
		/* Perform image warping into devX */
		nppiWarpPerspective_8u_P3R(srcPlanes, srcSize,     pkt.sizeIn, srcROI,
		                           dstPlanes, pkt.sizeOut, dstROI,     H,
		                           NPPI_INTER_LINEAR);
	}
	
	/* Close the memory handles. */
	cudaIpcCloseMemHandle(devX);
	cudaIpcCloseMemHandle(devY);
	
	/* Ping 1 byte back to the remote side to ACK */
	write(connCtx->sockFd, "\0", 1);
}

/**
 * Read one packet coming from the remote end of the connection.
 */

int  imgsConnReadPacket(GLOBAL_DATA* gData, CONN_CTX* connCtx, CONN_PKT* pkt){
	ssize_t bytesReadTotal = 0, bytesRead;
	
	/**
	 * Read fully a packet.
	 */
	
	do{
		bytesRead = read(connCtx->sockFd,
		                 (char*)pkt   + bytesReadTotal,
		                 sizeof(*pkt) - bytesReadTotal);
		
		if(bytesRead == 0){
			printf("Client closed socket on us.\n");
			return -1;
		}else if(bytesRead < 0){
			printf("Socket read error.\n");
			return -1;
		}else{
			bytesReadTotal += bytesRead;
		}
	}while(bytesReadTotal < sizeof(*pkt));
	
	/* Configure PRNG as requested */
	connCtx->xorshift128p[0] = pkt->x128ps0;
	connCtx->xorshift128p[1] = pkt->x128ps1;
	
	/* Return bytes read. */
	return bytesReadTotal;
}

/**
 * Select an image to warp, depending on the packet and iteration number.
 */

uint64_t imgsConnSelectImage(GLOBAL_DATA* gData,
                             CONN_CTX*    connCtx,
                             CONN_PKT*    pkt,
                             int          i){
	/**
	 * The master indices.
	 * TRAIN SET:     0:22499
	 * VALID SET: 22500:24999
	 */
	
	const uint64_t SPLIT[] = {
	    7051, 6570, 10600, 10690, 11746, 21248, 6305, 12852, 4507, 7458, 22682, 1234, 
	    13187, 2225, 1323, 22141, 13538, 4584, 24244, 15719, 21357, 23245, 17525, 6191, 
	    20107, 5988, 15940, 19042, 19370, 2354, 21093, 13279, 8214, 20949, 6992, 604, 
	    1423, 23436, 22176, 24003, 20833, 10277, 13939, 19444, 9769, 13520, 8423, 6897, 
	    533, 8753, 52, 22923, 7751, 20363, 16619, 9756, 3310, 4467, 13179, 13913, 3120, 
	    7068, 1206, 17070, 13396, 3735, 23042, 2778, 4313, 13271, 21530, 5114, 22944, 
	    9942, 22389, 17592, 17524, 24332, 10374, 12466, 14310, 19610, 14232, 24319, 
	    18930, 3962, 14353, 11298, 23294, 297, 24214, 1739, 12694, 15634, 23056, 7488, 
	    5762, 19884, 16661, 1803, 1, 13015, 23161, 2885, 5930, 1477, 13543, 8540, 13462, 
	    13007, 12229, 21089, 12581, 15371, 7721, 11091, 7672, 23932, 12892, 18025, 4389, 
	    2503, 12932, 7669, 19874, 351, 20094, 20141, 18490, 12239, 21665, 2770, 15477, 
	    9858, 15444, 12033, 9058, 5828, 22000, 16992, 5738, 5743, 11648, 14937, 2988, 
	    3439, 6105, 17601, 9353, 22711, 4494, 13345, 19820, 12861, 509, 18395, 21534, 
	    4283, 18661, 11565, 2902, 15076, 1177, 20600, 5412, 9753, 24330, 21206, 18875, 
	    1674, 17909, 18840, 20683, 4999, 5268, 14377, 21675, 505, 3692, 17846, 5863, 
	    21049, 14601, 24057, 18031, 19494, 17905, 14634, 538, 8452, 110, 23545, 8663, 
	    6195, 8300, 19234, 13984, 20547, 9458, 6234, 6825, 16842, 17078, 14118, 16832, 
	    13147, 20668, 2740, 2378, 3503, 16846, 18239, 9995, 17595, 11362, 3670, 24286, 
	    23180, 15359, 6576, 20187, 1122, 10056, 23129, 22816, 21890, 10643, 14214, 2144, 
	    8884, 18658, 2190, 4827, 14734, 23133, 8668, 3338, 24528, 10907, 12671, 8084, 
	    15950, 6043, 6009, 10818, 6854, 8025, 332, 20631, 2660, 13001, 12231, 11871, 
	    22707, 12366, 4170, 20584, 17878, 4623, 12241, 1925, 18206, 8721, 18049, 7868, 
	    19166, 21664, 20423, 1350, 4592, 19104, 17648, 1321, 6942, 10332, 14726, 11433, 
	    8202, 7938, 2410, 3687, 11901, 5701, 2954, 13100, 11642, 2353, 21328, 13881, 
	    7337, 116, 20919, 16898, 7891, 23206, 23155, 1050, 20879, 8397, 21923, 613, 
	    7806, 17246, 10630, 13222, 11961, 12950, 18602, 23200, 22446, 8872, 13691, 5451, 
	    6104, 12577, 3786, 20407, 6048, 23571, 158, 5276, 23981, 14467, 10109, 22029, 
	    8156, 18094, 6308, 15084, 21899, 24597, 3211, 5087, 11464, 17964, 12532, 7802, 
	    21555, 8717, 8235, 5812, 4475, 1760, 1145, 5648, 20645, 18246, 14181, 12141, 
	    869, 12881, 1170, 727, 2202, 20751, 3644, 6312, 19776, 19754, 16870, 6221, 6296, 
	    3907, 5294, 21943, 16611, 5330, 19107, 13834, 7843, 8874, 13362, 6959, 16607, 
	    17423, 14351, 21902, 1228, 20501, 17143, 9923, 21394, 5503, 18625, 5186, 258, 
	    11687, 16834, 20831, 4814, 9182, 23236, 12340, 21265, 15974, 11027, 18318, 
	    16848, 5765, 6193, 23604, 2043, 11454, 5432, 22362, 11303, 2585, 16061, 23040, 
	    17268, 2911, 22983, 23700, 8279, 16837, 12759, 23951, 19340, 16276, 18252, 8073, 
	    11479, 4366, 14317, 8355, 8302, 19916, 11267, 20442, 21558, 7285, 2512, 20025, 
	    17008, 6178, 4735, 4629, 10880, 22369, 1749, 18924, 2333, 24416, 19961, 22067, 
	    9168, 11921, 12226, 23543, 7702, 19357, 5468, 8207, 24303, 4983, 24736, 21278, 
	    19496, 12040, 11073, 2724, 5916, 6819, 7103, 4777, 16738, 23569, 17092, 24105, 
	    4903, 18389, 21449, 20795, 11902, 5522, 22670, 23130, 7547, 22909, 10157, 19218, 
	    1989, 6549, 7269, 23452, 23703, 24087, 7377, 14520, 23366, 9939, 17837, 19611, 
	    11787, 5576, 17585, 13189, 9640, 1916, 10744, 22946, 13136, 14438, 4972, 14347, 
	    8534, 15274, 14196, 10, 1828, 13476, 7886, 3487, 16477, 19233, 18934, 15765, 
	    16688, 610, 22490, 16920, 11631, 6951, 4118, 6534, 10496, 2861, 14557, 2952, 
	    19459, 7545, 708, 14571, 14412, 6620, 13845, 14881, 17647, 17168, 18543, 22875, 
	    546, 768, 3640, 2810, 21548, 15358, 5536, 12721, 21904, 14072, 737, 388, 16327, 
	    11266, 16732, 1404, 6748, 14488, 24107, 9491, 11134, 1362, 23608, 23934, 21695, 
	    6458, 7750, 16295, 7739, 2171, 5212, 22409, 10462, 17218, 12353, 22436, 15947, 
	    6377, 24565, 11571, 8208, 16905, 19436, 265, 19261, 5007, 17739, 2490, 2419, 
	    22672, 14418, 3255, 6824, 8442, 8751, 24509, 14407, 21799, 3108, 4863, 15062, 
	    19787, 19187, 23589, 7223, 20341, 23637, 4766, 1537, 22315, 22274, 17716, 3416, 
	    7170, 16597, 1220, 2356, 11033, 7944, 18970, 13654, 8763, 19132, 22614, 1316, 
	    6811, 24880, 8182, 21611, 165, 23476, 3954, 6414, 7016, 12289, 14518, 22732, 
	    23386, 20702, 21642, 1780, 16751, 5265, 20944, 1198, 17126, 8440, 19665, 18800, 
	    5788, 20221, 1949, 13419, 13696, 21322, 7910, 1715, 1207, 22590, 19765, 22769, 
	    6925, 19121, 14018, 10711, 11440, 17050, 19643, 14633, 10717, 17886, 2552, 
	    17745, 7326, 15109, 6046, 19257, 15450, 13097, 19168, 2231, 3265, 2499, 539, 
	    14497, 19857, 10309, 3745, 10063, 7554, 17099, 11607, 11494, 22525, 20105, 3691, 
	    16663, 6366, 14030, 6270, 6567, 17839, 3404, 22925, 13555, 12709, 9905, 6665, 
	    17572, 14610, 3327, 21745, 7501, 9506, 1958, 8737, 10926, 21582, 469, 12233, 
	    7376, 24942, 195, 12082, 19262, 23194, 19146, 1931, 16904, 12050, 15273, 20731, 
	    8035, 767, 2272, 14744, 20231, 12043, 18422, 6085, 5257, 6998, 15818, 143, 8612, 
	    8029, 13990, 15729, 10963, 1284, 17166, 17124, 9358, 10972, 23894, 11660, 8033, 
	    5697, 11304, 17434, 11308, 9210, 24221, 5086, 14653, 8326, 2987, 8674, 5487, 
	    18135, 2535, 1612, 15039, 400, 3183, 19946, 17656, 999, 9788, 9346, 3073, 11805, 
	    4434, 923, 19571, 6333, 11741, 6669, 17639, 882, 7559, 9759, 13900, 1737, 10103, 
	    9357, 4597, 9202, 23538, 7017, 17620, 5946, 2027, 17279, 22419, 5834, 11401, 
	    19897, 13106, 814, 24396, 10276, 15683, 10635, 8491, 13078, 16476, 22263, 10873, 
	    6140, 10012, 301, 3727, 13027, 24878, 4573, 6165, 20562, 5374, 21607, 4291, 
	    1762, 1846, 12133, 197, 11917, 9796, 2077, 11080, 12600, 23335, 20139, 19890, 
	    17509, 2244, 13468, 11297, 4934, 3266, 5253, 2830, 19497, 10893, 9502, 10428, 
	    5721, 16694, 8173, 18366, 12984, 3690, 9430, 1920, 7323, 5415, 17290, 8939, 
	    18253, 21714, 7933, 7088, 3006, 8337, 11513, 24893, 6938, 11399, 20861, 7449, 
	    13733, 11099, 3762, 3386, 14367, 17848, 16085, 22394, 3652, 16422, 18121, 9420, 
	    4023, 5774, 17244, 6062, 18505, 1065, 15238, 20422, 18414, 20166, 11295, 24570, 
	    12786, 2932, 13308, 12020, 24990, 6934, 3184, 18549, 8686, 24902, 15859, 20031, 
	    16318, 4400, 17548, 17804, 4247, 7522, 3994, 19137, 1184, 23980, 508, 19659, 
	    16234, 5651, 8926, 1496, 18338, 22042, 589, 14425, 18913, 24758, 11057, 17461, 
	    20052, 3752, 4991, 14528, 17987, 9723, 10227, 7949, 17132, 6913, 3582, 6227, 
	    750, 23818, 7126, 3243, 13210, 18964, 422, 17769, 5099, 7182, 8989, 24671, 
	    19583, 14223, 15320, 9238, 11809, 5797, 15136, 21040, 4451, 6815, 7074, 16938, 
	    4195, 7658, 14880, 23761, 14883, 4055, 22615, 20979, 16555, 23231, 9916, 15329, 
	    6899, 16622, 24164, 2669, 21097, 5698, 10024, 13196, 9734, 7107, 9355, 24265, 
	    13897, 16143, 2255, 20493, 15920, 14770, 24022, 22418, 1819, 22916, 18533, 
	    12952, 19576, 2621, 20339, 17920, 78, 16894, 10624, 20066, 23613, 20563, 16686, 
	    23054, 13501, 21432, 14648, 10754, 7113, 15972, 12029, 2191, 22620, 23497, 
	    16076, 16100, 9249, 10044, 12147, 8630, 23776, 16229, 4684, 9044, 19768, 946, 
	    10659, 20457, 22164, 2598, 17173, 1232, 22564, 8476, 14216, 12088, 2936, 3845, 
	    24520, 20809, 5479, 10608, 4382, 1732, 20966, 3853, 16460, 13567, 6513, 7211, 
	    8065, 6639, 20829, 13649, 18618, 9056, 14295, 818, 14682, 10521, 24434, 5303, 
	    21170, 14061, 2863, 19337, 10706, 2966, 10538, 8295, 13627, 23103, 7871, 1092, 
	    2588, 20667, 24001, 24133, 4017, 4301, 18962, 282, 15680, 23404, 24202, 1178, 
	    15793, 7452, 5485, 20346, 12787, 19246, 10045, 3825, 14754, 6762, 4197, 557, 
	    18576, 24307, 24079, 19453, 21183, 23128, 10892, 1919, 6937, 16386, 6461, 24651, 
	    14144, 10188, 2901, 10720, 13385, 4922, 7299, 12927, 12440, 7465, 7020, 11429, 
	    13662, 24735, 15489, 23816, 5446, 3465, 21878, 22837, 20384, 11205, 16088, 6336, 
	    10100, 16699, 12049, 699, 1391, 19802, 2105, 17398, 8813, 2706, 402, 17831, 
	    9742, 5029, 21133, 24636, 11164, 19003, 14381, 7319, 2894, 4264, 13421, 10301, 
	    18986, 21589, 24680, 3659, 6162, 4911, 13304, 8141, 8720, 14442, 15006, 473, 
	    1229, 2485, 4415, 11826, 22755, 11686, 13451, 12383, 15660, 9961, 16896, 10108, 
	    5108, 6376, 6680, 9468, 20401, 6895, 21550, 15152, 2172, 19560, 24279, 10495, 
	    12973, 15057, 20972, 12649, 9494, 18466, 10111, 11504, 20004, 22250, 10258, 
	    2057, 7940, 19642, 15736, 20498, 10004, 7748, 22559, 6340, 2605, 16689, 24977, 
	    2426, 19393, 19982, 13765, 11525, 12420, 16434, 16725, 18754, 3618, 13277, 
	    17229, 17136, 2631, 15010, 7681, 9869, 12561, 17282, 15769, 4371, 1153, 16859, 
	    4957, 14907, 24503, 6912, 5574, 15967, 8416, 19212, 10850, 1027, 2731, 9130, 
	    7726, 21078, 15401, 2306, 18138, 15589, 16886, 20415, 3094, 11946, 10580, 21191, 
	    4834, 341, 21198, 15943, 874, 21321, 17101, 9643, 20572, 8693, 4590, 1590, 
	    11541, 22391, 16054, 17619, 11619, 11150, 14136, 3866, 8204, 14228, 18304, 8623, 
	    15847, 24188, 20850, 7595, 13047, 7724, 20018, 13031, 585, 17685, 13940, 23828, 
	    17146, 19065, 2457, 21371, 21778, 21407, 24563, 14657, 8024, 14624, 24150, 
	    17992, 21149, 7455, 192, 5514, 21870, 5571, 3031, 14508, 7839, 22686, 20110, 
	    24780, 12814, 6904, 16415, 8638, 18060, 13307, 12062, 22197, 403, 12651, 24115, 
	    1574, 2676, 965, 21, 18955, 13238, 11543, 17479, 23574, 7960, 863, 16325, 18877, 
	    11104, 1838, 10441, 12845, 23705, 18644, 17114, 1022, 16044, 2474, 2153, 5149, 
	    18449, 2574, 7133, 24094, 6561, 8705, 4261, 24998, 21918, 23843, 24681, 635, 
	    17614, 13259, 1702, 22249, 24010, 21793, 21795, 18540, 18662, 23530, 20201, 
	    13200, 9493, 9574, 5100, 24803, 11922, 12896, 20259, 12762, 12054, 4087, 15103, 
	    23731, 3911, 5483, 12016, 19017, 4079, 15895, 11727, 17012, 13889, 16071, 12630, 
	    10723, 11801, 19307, 4340, 22302, 8002, 24907, 13101, 18584, 11347, 24705, 4974, 
	    12875, 24413, 10445, 12363, 4541, 12834, 5098, 2092, 22101, 15763, 11806, 13042, 
	    18266, 11156, 9668, 15098, 8793, 8895, 5696, 10826, 18229, 2302, 5117, 18437, 
	    3539, 9014, 2790, 21733, 8907, 82, 5617, 1160, 4131, 4809, 13195, 6982, 24099, 
	    11556, 24862, 19004, 7889, 23639, 12248, 16378, 15431, 12979, 14780, 21959, 
	    20582, 3885, 2618, 13380, 9561, 17694, 13328, 23336, 24181, 5830, 4003, 18769, 
	    18647, 4613, 7327, 14847, 14623, 2160, 6738, 10546, 1031, 10673, 8061, 426, 
	    3720, 6395, 14218, 11390, 11070, 9248, 4048, 571, 14270, 3171, 5607, 18235, 
	    13428, 900, 18344, 19319, 19953, 22694, 84, 7920, 9156, 6518, 198, 17029, 20905, 
	    20246, 19876, 11372, 3742, 7084, 12017, 19999, 1396, 4693, 18799, 7772, 21867, 
	    13623, 22506, 10974, 21862, 6616, 5208, 11904, 13650, 3522, 2918, 6173, 10391, 
	    17250, 9912, 11247, 11252, 15232, 14081, 19790, 9620, 2152, 11539, 21417, 9290, 
	    12145, 19771, 5836, 18316, 6767, 11963, 6035, 3924, 5502, 11076, 7439, 15640, 
	    18659, 5956, 10614, 5448, 7630, 16487, 5241, 10328, 19929, 1884, 12542, 19675, 
	    5495, 23154, 23092, 5544, 7237, 23148, 3033, 16231, 5802, 23863, 22323, 19522, 
	    1421, 4021, 22994, 14380, 5756, 13323, 18209, 24085, 7395, 3372, 11398, 17976, 
	    22025, 1820, 18663, 8095, 9167, 8376, 11823, 14643, 17981, 22680, 9645, 24763, 
	    13547, 13933, 12250, 4466, 14865, 13997, 20769, 638, 7322, 24119, 1665, 9835, 
	    14293, 6783, 886, 18453, 2510, 14905, 13527, 4424, 3070, 12220, 8579, 20311, 
	    24451, 19707, 10980, 15789, 18013, 14082, 797, 16665, 11463, 24601, 13370, 
	    23192, 2367, 24436, 9644, 23805, 13242, 15217, 19277, 4655, 12572, 6148, 655, 
	    11882, 2449, 15575, 91, 1369, 12847, 2291, 22196, 12522, 23993, 2081, 1338, 
	    22813, 19777, 13010, 18697, 3200, 23809, 12902, 14156, 12267, 6662, 23254, 
	    19631, 9748, 9402, 2334, 8930, 2516, 9974, 7565, 21602, 3131, 311, 19588, 19419, 
	    22861, 8512, 11130, 19066, 1675, 8083, 23911, 11013, 20990, 2210, 15246, 4511, 
	    1964, 13719, 12308, 8941, 14979, 12260, 18811, 23693, 16320, 13618, 18290, 1062, 
	    15711, 3751, 5524, 11724, 3000, 19392, 5630, 10073, 21820, 16045, 10189, 9165, 
	    24102, 2872, 20159, 4022, 8701, 4455, 20274, 3519, 8128, 17119, 19225, 848, 
	    21988, 10768, 24158, 1556, 20469, 9397, 19960, 23954, 8206, 7718, 16035, 14720, 
	    2053, 17222, 7199, 1148, 18874, 21243, 11233, 19096, 22155, 21583, 20216, 9727, 
	    13232, 2751, 4106, 11953, 9951, 6468, 5609, 10223, 12555, 13588, 23047, 3105, 
	    7105, 15624, 12680, 10611, 24833, 711, 7641, 9636, 6836, 14040, 4375, 21388, 
	    20336, 16741, 9962, 4767, 3741, 5532, 3067, 20978, 12352, 8328, 21021, 23345, 
	    5684, 19829, 15193, 21120, 14422, 20229, 5868, 8434, 23492, 15458, 11523, 8681, 
	    19794, 10025, 17460, 22295, 722, 14391, 7942, 9071, 6175, 23633, 2447, 24852, 
	    14885, 6290, 20255, 11359, 2376, 12858, 10446, 18180, 2943, 17989, 12718, 2728, 
	    4673, 7541, 10036, 910, 20426, 3761, 12094, 5419, 17849, 1011, 14728, 19471, 
	    5491, 23120, 3923, 7706, 671, 13294, 3636, 14389, 17390, 8087, 24296, 21175, 
	    24316, 899, 9851, 16792, 15617, 392, 17200, 3891, 3007, 3515, 15018, 11734, 
	    13930, 11675, 13478, 21662, 2248, 2874, 747, 10867, 2711, 17157, 12355, 20723, 
	    17512, 14443, 10847, 140, 5006, 1331, 3844, 13750, 6553, 12518, 1230, 21860, 
	    23575, 9098, 11222, 19639, 24104, 9501, 17081, 15853, 3224, 21618, 1707, 11145, 
	    12327, 14193, 10669, 11511, 21381, 4797, 3594, 14388, 8626, 15245, 11394, 11030, 
	    662, 18175, 791, 756, 5266, 8286, 12167, 10584, 4031, 10618, 22318, 9812, 20033, 
	    15908, 7956, 12822, 18123, 967, 1878, 19269, 18747, 4192, 9423, 17818, 21458, 
	    15814, 9201, 3965, 15024, 4926, 13130, 5213, 418, 21776, 11090, 15604, 14179, 
	    22599, 22572, 6331, 23736, 6334, 11320, 2910, 21615, 784, 14663, 22898, 16040, 
	    6083, 23501, 22443, 14563, 24466, 8933, 3486, 2661, 15512, 21126, 16591, 5556, 
	    21113, 51, 20371, 19688, 16552, 12711, 21210, 8001, 3862, 22841, 22542, 21963, 
	    1821, 18764, 10042, 10742, 20439, 12693, 21470, 7174, 4634, 5131, 6087, 24473, 
	    1649, 6345, 10640, 8803, 19109, 4420, 21273, 64, 2647, 15608, 16091, 22963, 
	    21326, 5940, 15733, 21757, 8680, 15652, 6232, 13965, 383, 23058, 15324, 19451, 
	    3403, 22717, 2405, 23137, 10581, 24429, 21696, 7200, 24971, 2408, 22915, 17559, 
	    23477, 14213, 5473, 4249, 4146, 5626, 7436, 18279, 15257, 18250, 20794, 1772, 
	    1651, 4292, 102, 15777, 18630, 8205, 23318, 20154, 15788, 15443, 8997, 7049, 
	    18926, 13402, 9789, 19386, 23059, 9309, 23923, 2864, 7009, 13812, 14969, 2454, 
	    22473, 13952, 24717, 901, 12647, 1286, 3319, 7482, 22966, 21228, 19620, 4182, 
	    10013, 15803, 23858, 24406, 13107, 24472, 8, 13448, 20306, 4293, 14398, 4014, 
	    2854, 13039, 20580, 14096, 2012, 22071, 12811, 2639, 12464, 12382, 15435, 13469, 
	    2121, 8734, 18653, 4552, 17833, 19447, 17177, 17467, 6472, 12849, 15023, 3361, 
	    20361, 4943, 23897, 16281, 20644, 14089, 20750, 828, 9217, 18286, 24138, 11936, 
	    2839, 6866, 11548, 17255, 11202, 6877, 6008, 4116, 8932, 9189, 19028, 20285, 
	    6214, 23648, 21946, 5573, 12187, 4468, 10122, 18367, 11137, 5164, 4823, 2245, 
	    9926, 12535, 10943, 20240, 3935, 6352, 2240, 23586, 4355, 18674, 13156, 13099, 
	    15072, 8776, 17068, 21382, 13491, 24410, 15456, 24008, 746, 23187, 12336, 16508, 
	    20838, 16160, 18667, 10629, 24130, 4839, 18071, 14922, 5367, 4670, 1015, 1299, 
	    18616, 8049, 6801, 4070, 23089, 15884, 14057, 15260, 2443, 7478, 9738, 3409, 
	    23713, 17347, 22089, 24060, 6747, 8683, 6832, 6940, 19201, 16184, 14102, 22987, 
	    161, 6441, 11177, 14275, 23788, 17431, 17538, 23539, 18358, 6368, 24727, 5577, 
	    664, 12194, 2065, 7629, 2362, 8368, 7028, 23395, 12558, 3110, 8473, 10941, 
	    11908, 23625, 3049, 16550, 10728, 131, 18585, 24540, 16913, 1256, 18026, 11780, 
	    12790, 5543, 22205, 11019, 274, 9809, 13706, 15150, 23158, 21763, 12298, 18673, 
	    21266, 17211, 21295, 5910, 1043, 10871, 22200, 2134, 19818, 3543, 6711, 14129, 
	    10264, 6691, 22739, 7116, 20610, 4587, 18489, 17578, 13814, 17664, 11975, 14548, 
	    18827, 617, 4442, 5027, 11200, 24366, 17563, 5596, 22371, 12659, 8299, 14227, 
	    16585, 18445, 1115, 10617, 23892, 1622, 1680, 22336, 6941, 13755, 16995, 24999, 
	    20714, 4341, 17260, 23669, 8618, 21853, 2284, 1805, 8085, 12245, 10381, 17102, 
	    904, 17996, 3052, 9615, 7884, 12604, 2529, 16294, 22057, 6386, 12971, 23706, 
	    7835, 19834, 16355, 15717, 16541, 16562, 9873, 12664, 4115, 15755, 6224, 24923, 
	    17309, 16534, 14477, 20524, 11828, 22949, 18831, 5355, 12304, 5255, 4065, 24502, 
	    1851, 24024, 24749, 4641, 24786, 8289, 21105, 3446, 6171, 2678, 21405, 9664, 
	    15882, 2150, 8178, 15394, 13168, 17386, 8031, 16203, 15095, 13688, 11387, 22384, 
	    3843, 17338, 3860, 5291, 9794, 7890, 21404, 1988, 7796, 290, 3352, 4727, 24263, 
	    14219, 8901, 2905, 14918, 24412, 9642, 14257, 10373, 16681, 23303, 23871, 3784, 
	    21119, 10792, 20268, 11419, 21401, 16390, 9527, 14890, 18081, 10238, 15679, 
	    24293, 20801, 13624, 13718, 22806, 11402, 24625, 23827, 17150, 8637, 17044, 
	    24775, 8220, 11551, 4449, 4560, 2278, 2772, 20378, 9963, 4101, 13008, 10715, 
	    23874, 22035, 17916, 3304, 23741, 10993, 13401, 19862, 2583, 24914, 15936, 
	    24831, 19773, 17764, 23527, 13376, 22276, 24704, 15258, 3426, 20109, 20811, 
	    16149, 24543, 19203, 24584, 7505, 12584, 22839, 7801, 20745, 15732, 17274, 
	    11700, 10123, 5860, 10107, 12391, 4123, 1443, 2796, 16279, 11593, 21385, 17772, 
	    9123, 17038, 19607, 114, 21203, 22169, 7828, 1911, 4032, 6155, 20114, 9940, 
	    1475, 7453, 4446, 17863, 16117, 628, 7701, 4803, 20925, 7863, 20287, 24730, 
	    2748, 10330, 18149, 732, 14872, 1742, 6539, 7973, 14283, 9932, 151, 745, 5964, 
	    2895, 13169, 20963, 10739, 13713, 14637, 24571, 4013, 24943, 9872, 16826, 10798, 
	    5878, 19013, 5200, 15206, 13851, 23257, 17359, 16820, 17265, 21807, 4703, 3566, 
	    9682, 2028, 21232, 4318, 6069, 18341, 623, 12456, 10260, 24438, 23165, 13059, 
	    7885, 5679, 10080, 7065, 11131, 18862, 8090, 8980, 3781, 13884, 1823, 16226, 
	    23260, 2492, 20655, 9660, 846, 10927, 17110, 4508, 23050, 2927, 5001, 14152, 
	    5385, 16488, 22271, 16909, 19438, 18539, 2052, 9199, 17508, 16726, 6309, 3443, 
	    4688, 2613, 5848, 24398, 17787, 2476, 13995, 4396, 11861, 5713, 15889, 15308, 
	    19664, 8123, 8859, 20716, 16202, 11677, 710, 9257, 6763, 22016, 6047, 10299, 
	    15471, 128, 23064, 15898, 2364, 19727, 2166, 24752, 4333, 21753, 13276, 7191, 
	    17768, 1867, 10086, 2350, 16438, 2075, 6485, 689, 18385, 8844, 5595, 16590, 
	    24360, 4859, 13894, 6471, 22106, 17549, 21620, 6653, 13641, 23391, 4925, 19827, 
	    18260, 17645, 23558, 22551, 9871, 22639, 10828, 23505, 4665, 8293, 10656, 17599, 
	    21815, 2636, 8441, 14993, 13316, 2866, 12751, 24147, 2798, 4236, 12102, 2122, 
	    24773, 11421, 6353, 24041, 24670, 24145, 3218, 23532, 11909, 14120, 4961, 988, 
	    19174, 499, 3631, 16772, 22052, 21895, 13579, 7118, 19268, 9338, 4155, 107, 
	    23964, 21725, 7575, 14300, 8089, 15914, 6995, 22021, 11093, 23212, 3814, 23744, 
	    2260, 1132, 22501, 16480, 16885, 22413, 16633, 7110, 3967, 12609, 15479, 18473, 
	    9022, 11215, 20329, 5837, 2159, 2416, 24390, 14101, 5320, 1753, 15236, 13406, 
	    4718, 891, 18508, 8081, 6522, 9630, 2628, 23587, 12961, 6770, 19737, 17336, 
	    6511, 13246, 10054, 5635, 2156, 13470, 10093, 23561, 10682, 1497, 9404, 17027, 
	    23466, 15123, 2114, 20712, 19909, 21022, 19548, 16242, 8486, 73, 7216, 17300, 
	    2992, 24905, 14526, 8424, 22482, 5467, 7353, 16187, 22030, 19410, 11678, 10490, 
	    21969, 10550, 12059, 5069, 12024, 12821, 11711, 13840, 13721, 12594, 11173, 
	    1598, 10145, 12566, 16285, 22932, 24241, 21785, 20340, 20223, 17287, 13014, 
	    23795, 11564, 4538, 545, 18403, 6381, 21488, 9095, 3053, 21305, 21673, 18734, 7, 
	    2622, 4543, 17565, 8676, 7071, 8834, 9445, 356, 22577, 13314, 4061, 11601, 
	    15209, 4562, 1555, 17339, 14058, 8197, 10782, 2913, 7906, 8000, 2539, 1414, 
	    19732, 12034, 1063, 18681, 11736, 12486, 6207, 5575, 412, 13539, 17774, 10455, 
	    22826, 4481, 10112, 23016, 14277, 24848, 5657, 14305, 10977, 19133, 10610, 
	    19341, 24433, 3686, 4807, 15783, 21143, 12160, 8149, 18457, 12637, 12974, 1860, 
	    10091, 14100, 12077, 19472, 9602, 3798, 6094, 16998, 12841, 19305, 15361, 19525, 
	    8928, 17366, 13138, 17844, 4880, 22880, 14822, 21654, 22307, 22512, 3370, 23319, 
	    17426, 17438, 19956, 24037, 7770, 2329, 1810, 8451, 20981, 12316, 23716, 19067, 
	    15857, 24023, 20857, 10150, 16685, 2307, 7368, 1233, 15476, 12853, 6295, 23013, 
	    11224, 17927, 10898, 336, 15464, 921, 2096, 10014, 20175, 861, 11381, 6488, 
	    7531, 16989, 22162, 9470, 15645, 22624, 5336, 20906, 14302, 22632, 24397, 11546, 
	    18696, 21501, 13918, 14691, 6853, 21238, 8945, 12515, 18211, 15669, 8950, 477, 
	    18352, 20526, 2129, 13526, 1584, 4025, 3241, 939, 11941, 4046, 2298, 22404, 
	    3039, 21070, 19347, 2526, 2593, 19248, 11870, 18976, 3948, 634, 23413, 1514, 
	    20352, 21767, 3959, 17389, 3375, 2082, 16210, 14533, 4644, 17682, 9180, 18106, 
	    11982, 2070, 9717, 16547, 9591, 23974, 23041, 5315, 1216, 3542, 8057, 20528, 
	    15173, 22119, 19686, 13943, 24253, 19654, 8343, 1428, 23638, 13609, 13669, 2697, 
	    12702, 10565, 24787, 24750, 15396, 2379, 18573, 859, 10162, 15099, 14896, 6060, 
	    3414, 17207, 24066, 9016, 24627, 867, 21052, 20386, 1120, 24739, 2118, 3068, 
	    239, 14909, 19927, 23916, 16059, 9461, 11758, 18749, 20748, 22967, 5159, 5876, 
	    6850, 16362, 9998, 7365, 2881, 4681, 23953, 9301, 24335, 1136, 8931, 13065, 
	    5549, 20122, 9387, 2010, 8464, 7258, 4323, 6863, 12403, 21347, 20996, 1553, 
	    17732, 19934, 18716, 2115, 13949, 3638, 13731, 20653, 21241, 8908, 11643, 3250, 
	    17588, 10761, 22859, 20656, 23925, 6145, 12414, 880, 14446, 16385, 18917, 7558, 
	    16382, 1942, 3508, 23612, 5222, 8092, 22072, 10369, 16610, 13383, 1605, 13819, 
	    14010, 24116, 3857, 8086, 1186, 21376, 20494, 9339, 16156, 16358, 16753, 19911, 
	    24141, 10371, 22774, 2375, 7067, 7109, 18758, 16092, 20089, 17776, 5262, 7492, 
	    6652, 10185, 14714, 16212, 8670, 7714, 518, 11926, 21525, 23520, 16674, 10511, 
	    236, 19423, 2325, 23422, 12734, 6578, 16768, 20385, 663, 11380, 16205, 4759, 
	    1026, 15012, 20998, 7674, 3626, 1974, 5285, 3209, 12530, 3879, 20011, 13012, 
	    21926, 11538, 8631, 12292, 21115, 9195, 15522, 6954, 12568, 4532, 10357, 1137, 
	    433, 8915, 20873, 1263, 17443, 15170, 22270, 3450, 12924, 13463, 24217, 1577, 
	    13545, 9439, 10501, 3079, 22181, 11469, 1800, 8536, 24810, 321, 12197, 20102, 
	    4483, 16361, 20650, 3767, 15530, 14261, 427, 14137, 5855, 817, 23310, 24798, 
	    8262, 2326, 19329, 11115, 15761, 21446, 23873, 7715, 3880, 7895, 8992, 7899, 
	    8256, 22387, 18257, 15587, 14146, 8828, 7602, 3669, 3588, 9533, 19149, 14171, 
	    13219, 1996, 12685, 18942, 6978, 19959, 14202, 922, 3657, 24457, 12279, 13596, 
	    11488, 10589, 20657, 9919, 20082, 9051, 16147, 3045, 10902, 7358, 6307, 9025, 
	    11373, 4185, 16797, 20170, 24973, 15301, 10548, 15055, 18880, 9874, 7268, 11095, 
	    6928, 12319, 13386, 13525, 18380, 11, 20460, 6446, 5300, 5082, 15725, 20812, 
	    5378, 208, 3074, 4353, 1887, 11206, 7901, 593, 8366, 10464, 7014, 8564, 6292, 
	    9161, 12429, 23845, 5061, 340, 19536, 12529, 11749, 9542, 12740, 24497, 15204, 
	    19502, 16698, 7280, 7551, 4596, 19304, 3064, 1254, 9411, 3219, 16319, 2725, 
	    1089, 14289, 22150, 9001, 6393, 8280, 17730, 4053, 23972, 3693, 22359, 24160, 
	    18879, 17799, 16346, 20669, 12722, 23592, 20967, 23959, 15822, 7831, 17073, 
	    23240, 24711, 6031, 4768, 21292, 7021, 8322, 4843, 20024, 24182, 2083, 12695, 
	    7459, 17379, 9829, 5047, 17453, 1697, 11077, 15766, 12941, 24919, 12079, 3800, 
	    21386, 20198, 3234, 529, 6893, 17720, 9295, 9892, 4842, 6891, 20158, 4526, 1188, 
	    7120, 24622, 3492, 16012, 12377, 24906, 786, 11490, 10262, 13549, 14426, 5044, 
	    10426, 14858, 19550, 18218, 5808, 23577, 5440, 4806, 8179, 18596, 4207, 15423, 
	    22290, 16976, 17495, 7970, 23955, 5819, 15970, 10609, 1758, 2109, 536, 13702, 
	    2165, 14942, 11026, 21739, 24867, 200, 14988, 7406, 5144, 3583, 14580, 4307, 
	    10448, 2162, 1326, 16518, 13904, 13764, 21411, 12650, 1123, 249, 19182, 22989, 
	    2961, 20545, 18402, 8752, 2104, 19975, 14998, 9361, 24599, 9852, 24595, 8038, 
	    19118, 18821, 7313, 4700, 22964, 2838, 24077, 6837, 12273, 3655, 21648, 1280, 
	    5359, 18077, 16839, 18481, 23034, 6911, 1565, 10293, 17643, 4121, 23973, 5979, 
	    3374, 11542, 5585, 781, 19219, 13327, 1375, 15706, 4662, 3632, 3899, 11945, 
	    5873, 13820, 14373, 19162, 326, 23084, 207, 12455, 9554, 13741, 21042, 9836, 
	    12592, 23430, 17381, 16057, 23896, 7315, 4156, 23122, 16770, 16700, 17019, 
	    11356, 15834, 17373, 24350, 10632, 21028, 5794, 17967, 6606, 13942, 11451, 1985, 
	    22477, 723, 1705, 16593, 23746, 9382, 24237, 4595, 3697, 5858, 11379, 7556, 65, 
	    10392, 3303, 12269, 7078, 8388, 446, 5843, 24834, 22805, 17684, 14203, 6808, 
	    6688, 19812, 14626, 23513, 18529, 15048, 21884, 718, 14743, 18203, 3528, 14673, 
	    14147, 6930, 17891, 15646, 10139, 14046, 3621, 4993, 1850, 8329, 21567, 20637, 
	    3685, 6634, 12735, 12280, 9185, 7905, 3641, 20844, 16643, 15322, 3920, 18966, 
	    13201, 22812, 15218, 15309, 6849, 23183, 10819, 13225, 13319, 942, 11129, 4947, 
	    18297, 18409, 17167, 20591, 24839, 4503, 5410, 20243, 24929, 5173, 11109, 16729, 
	    3244, 11845, 16807, 17317, 4963, 10315, 19336, 3452, 5238, 21993, 19483, 7360, 
	    1167, 7992, 20084, 11383, 17704, 18941, 6477, 3933, 21285, 4881, 21128, 20608, 
	    18611, 17149, 15268, 17598, 11269, 3214, 15994, 17482, 5924, 14536, 13084, 
	    14835, 19970, 21846, 19678, 9109, 12276, 20331, 8446, 7278, 21229, 19936, 99, 
	    23956, 1578, 12538, 21716, 4547, 6805, 19791, 24282, 2404, 13412, 24159, 4184, 
	    10037, 19943, 6586, 3533, 23218, 10197, 7799, 6613, 11665, 2263, 8875, 22445, 
	    13876, 16429, 20752, 2102, 12475, 7091, 15331, 12591, 9444, 2886, 23327, 10323, 
	    19382, 5665, 6841, 12333, 19470, 8527, 11293, 4361, 22260, 24033, 7225, 11353, 
	    17049, 5974, 21580, 11528, 5526, 14085, 5925, 19334, 5542, 22124, 11838, 3462, 
	    16544, 12392, 9344, 6365, 23727, 16119, 20097, 5304, 20771, 23461, 15997, 4988, 
	    14607, 1116, 15381, 6141, 10050, 15666, 3710, 2934, 18782, 16066, 6107, 1133, 
	    6584, 17322, 1357, 4093, 12044, 3271, 2001, 10192, 11116, 15379, 19356, 118, 
	    24771, 21553, 19705, 10131, 8439, 11311, 9864, 19029, 19325, 2555, 11035, 4168, 
	    11168, 12324, 21786, 14750, 13235, 23072, 12379, 19395, 4660, 13137, 13173, 
	    24273, 23987, 23376, 22434, 23153, 13656, 14236, 14741, 16695, 12215, 15709, 
	    16810, 18201, 12069, 5515, 20894, 1641, 14416, 22328, 2753, 2025, 13907, 11482, 
	    15928, 339, 24829, 19781, 10694, 20597, 16893, 18222, 944, 21881, 13017, 976, 
	    19800, 5578, 10021, 21554, 16789, 12092, 2221, 5052, 16400, 2812, 22939, 21461, 
	    22917, 17890, 8955, 19293, 7686, 5475, 9438, 21171, 2739, 8310, 11989, 4625, 
	    24928, 4206, 352, 24660, 11754, 21391, 10068, 8248, 22752, 13668, 20006, 5209, 
	    1957, 4029, 3637, 22412, 23950, 22468, 11604, 10385, 21889, 8664, 3538, 18687, 
	    23607, 4852, 8075, 24357, 18943, 19489, 18342, 7882, 20913, 6612, 15077, 24255, 
	    20935, 19207, 2327, 1444, 22857, 24431, 9379, 17471, 19452, 3060, 10951, 11382, 
	    23424, 18792, 1305, 16180, 18176, 16073, 13184, 8689, 2849, 14550, 11669, 23763, 
	    15343, 16945, 3529, 4504, 10329, 10303, 19101, 1462, 14531, 17912, 12073, 5324, 
	    10933, 13410, 8568, 6011, 20309, 1415, 5243, 12041, 3390, 16856, 6631, 2698, 
	    21573, 2384, 2909, 17077, 9520, 20802, 24544, 7745, 2898, 24135, 16815, 5298, 
	    4411, 3499, 15235, 7335, 2525, 13785, 12356, 7981, 3993, 20929, 22656, 1179, 
	    22475, 13067, 7421, 19091, 14439, 21090, 21063, 11555, 16472, 21756, 6438, 
	    11497, 10810, 21976, 11028, 7069, 17058, 17009, 10010, 15104, 19242, 11092, 
	    16953, 1168, 3808, 4282, 16833, 10254, 14617, 24617, 23508, 3544, 847, 22758, 
	    17371, 23450, 7384, 10895, 7134, 24918, 5140, 32, 11143, 9148, 19547, 1987, 
	    23930, 18412, 4273, 13180, 17713, 6480, 4290, 12378, 20754, 6487, 2266, 1706, 
	    1575, 4499, 18182, 10832, 15054, 14753, 4177, 16018, 23264, 771, 8251, 17351, 
	    5424, 12487, 20230, 606, 11018, 7012, 27, 9124, 22076, 789, 5822, 9254, 12774, 
	    19740, 7964, 13621, 1061, 12891, 23573, 6436, 22108, 24534, 24685, 1954, 17581, 
	    13262, 15135, 1166, 6407, 24868, 16777, 22144, 12046, 2970, 19300, 12504, 14820, 
	    14263, 16606, 18984, 23572, 21971, 657, 20959, 10348, 17510, 24088, 17447, 
	    23812, 24297, 1781, 11178, 3749, 726, 10411, 12198, 20404, 8641, 3992, 23913, 
	    15975, 24606, 5234, 20587, 11644, 23782, 2658, 2421, 15250, 17267, 2423, 20593, 
	    11549, 7474, 23407, 15848, 436, 15125, 6949, 9412, 3627, 4626, 8479, 823, 18320, 
	    18411, 9626, 17292, 23860, 2938, 6341, 23118, 8661, 7442, 22781, 23820, 15509, 
	    24930, 9585, 5301, 10400, 14582, 24940, 17100, 22799, 15366, 18665, 5427, 3739, 
	    22495, 830, 4398, 16401, 15577, 22467, 13862, 8072, 15046, 20772, 24734, 14779, 
	    6839, 18558, 15628, 8115, 19352, 16589, 15807, 8518, 1180, 23334, 4849, 10153, 
	    20989, 15290, 8482, 21011, 16985, 277, 6716, 11951, 16710, 16660, 17545, 21743, 
	    4354, 9252, 15858, 605, 2189, 10641, 2632, 14067, 20883, 20289, 23224, 17795, 
	    3089, 4394, 4840, 17039, 14777, 15070, 15092, 9532, 10553, 15409, 6969, 21039, 
	    5465, 8840, 5196, 20847, 9721, 21574, 19992, 757, 491, 19405, 10163, 4519, 
	    12681, 1815, 679, 21685, 9972, 21936, 24118, 24532, 237, 15649, 16629, 3768, 
	    5710, 9480, 20128, 25, 19102, 4450, 15724, 15886, 22724, 1277, 2968, 24762, 
	    14919, 9854, 22498, 22380, 23696, 3996, 4363, 13397, 2771, 21483, 4314, 18622, 
	    7415, 13512, 1960, 24200, 3973, 17303, 17355, 15336, 1604, 22370, 18851, 12654, 
	    3957, 2939, 23872, 8947, 21837, 1190, 22345, 14846, 20700, 18824, 15341, 18954, 
	    16314, 19090, 14512, 6418, 317, 15133, 14559, 2867, 16, 6856, 5939, 18484, 
	    12510, 1606, 16169, 15222, 19389, 4039, 7086, 14116, 17364, 11176, 11377, 20266, 
	    3307, 5049, 21324, 14490, 19387, 2386, 13867, 22942, 13695, 9765, 8814, 16780, 
	    13083, 13774, 4240, 12484, 11927, 23848, 9810, 7394, 11574, 8589, 18038, 4611, 
	    20730, 7375, 16471, 24813, 13054, 9208, 8409, 8922, 16052, 16287, 19672, 18715, 
	    4190, 637, 13321, 18324, 105, 21868, 19513, 1534, 6896, 3180, 17855, 23426, 
	    10661, 3457, 19949, 3511, 24844, 24559, 9207, 5841, 11341, 21894, 5170, 7773, 
	    3549, 8495, 22729, 20310, 18965, 831, 69, 1432, 18255, 4650, 3816, 74, 21858, 
	    8767, 14341, 16243, 14337, 1735, 12109, 10416, 3360, 17120, 8869, 12704, 18448, 
	    11701, 7566, 7867, 6572, 23725, 13211, 2352, 23223, 9373, 609, 13311, 14385, 
	    825, 7959, 23291, 22451, 24494, 18536, 3575, 6503, 1965, 17810, 7232, 22015, 
	    23629, 5193, 3024, 20470, 10140, 18023, 24800, 21740, 19037, 7359, 10605, 19034, 
	    620, 23237, 18816, 428, 1983, 444, 964, 19864, 23287, 5639, 4142, 5839, 653, 
	    14280, 3571, 1310, 3460, 17079, 7295, 15959, 16809, 6637, 12179, 19758, 4309, 
	    7056, 16324, 9415, 4956, 4153, 8659, 22096, 21027, 23615, 16230, 1406, 454, 
	    11948, 19940, 21703, 20176, 11039, 21771, 24339, 18150, 18441, 24967, 7124, 
	    4886, 19099, 7783, 19076, 9053, 1549, 22171, 7147, 22241, 10288, 762, 1751, 
	    7197, 20445, 24405, 1671, 18995, 5753, 2596, 3162, 4906, 10006, 6813, 631, 
	    21633, 16306, 24260, 18822, 7819, 9368, 17589, 13288, 550, 5453, 11637, 5730, 
	    8139, 15093, 2234, 17040, 16642, 19081, 11114, 24048, 16181, 14567, 24083, 
	    19294, 4901, 1223, 4295, 11172, 517, 217, 15689, 4321, 22771, 11199, 23166, 
	    14569, 14358, 21850, 20598, 6033, 17329, 4488, 23826, 23031, 17249, 23511, 241, 
	    9688, 13725, 10076, 14956, 7909, 1103, 23914, 13723, 19197, 15486, 5798, 16369, 
	    20403, 9062, 8234, 10083, 4909, 21236, 16531, 23623, 1470, 12314, 24209, 9000, 
	    4219, 3547, 3174, 23846, 23479, 23806, 17494, 19769, 20628, 11652, 7900, 9159, 
	    14382, 24620, 24640, 3447, 8022, 21683, 10695, 4453, 7605, 7936, 10297, 6840, 
	    2551, 1673, 4161, 13044, 13597, 13966, 13598, 16761, 19743, 3647, 21645, 5116, 
	    21300, 16745, 3473, 1855, 19141, 23557, 17094, 2543, 12459, 2277, 8851, 2058, 
	    13657, 22177, 17082, 14340, 7156, 17284, 4894, 14547, 21102, 17748, 24338, 
	    15653, 14054, 18186, 2201, 22137, 19689, 647, 1221, 335, 11609, 9878, 7179, 
	    3574, 17984, 14574, 465, 11779, 4050, 19702, 24076, 16737, 18768, 24958, 11333, 
	    12322, 14055, 2116, 23942, 18931, 12099, 2530, 315, 10135, 21422, 21758, 19308, 
	    22288, 22792, 413, 17217, 1509, 5203, 23364, 14901, 3179, 3256, 22496, 2641, 
	    7450, 18985, 22867, 12165, 1210, 10459, 9452, 15958, 18717, 1334, 16828, 21136, 
	    11148, 58, 5971, 17197, 20451, 4522, 11007, 7685, 24722, 9724, 6457, 13291, 
	    23351, 4256, 18960, 24464, 3620, 3163, 12492, 8550, 7611, 11981, 24874, 10221, 
	    5905, 2303, 18036, 14206, 19725, 24828, 16927, 24614, 24939, 16843, 14642, 
	    15491, 9139, 5689, 8471, 14078, 10396, 10628, 15326, 7577, 18369, 13671, 12151, 
	    15531, 23774, 4352, 24650, 4404, 24486, 15702, 19278, 7399, 23580, 23263, 1467, 
	    14984, 429, 9556, 24005, 24143, 23301, 12149, 22113, 3714, 13919, 6179, 5541, 
	    23833, 8585, 19135, 22919, 23702, 24682, 20961, 16353, 9162, 11391, 9141, 15700, 
	    10397, 9603, 4658, 836, 19709, 18535, 3387, 8313, 8116, 5393, 12256, 10936, 
	    1215, 22759, 16425, 10606, 20366, 9004, 1526, 15139, 14545, 17788, 11059, 10812, 
	    12413, 12311, 18240, 7433, 16851, 3744, 22358, 2458, 3578, 16727, 4512, 5150, 
	    13575, 20273, 16421, 565, 9247, 13230, 23815, 3983, 8750, 23738, 8318, 3606, 
	    21510, 22454, 4904, 22516, 22982, 3748, 955, 5872, 6874, 18393, 17067, 24978, 
	    24345, 19996, 19987, 4096, 2717, 5505, 22248, 1448, 6750, 8624, 211, 13344, 
	    10467, 16926, 18300, 16955, 14566, 14387, 23667, 21909, 9994, 13348, 21230, 
	    12104, 22003, 2016, 16854, 7114, 1025, 1187, 17951, 23255, 23733, 1255, 5652, 
	    22157, 15462, 8615, 7349, 879, 11956, 1558, 4798, 8013, 14143, 7416, 10821, 
	    12574, 20087, 15467, 15987, 19577, 9911, 23259, 22811, 9163, 15461, 16245, 
	    13351, 12691, 7624, 22509, 11322, 1313, 10660, 20467, 16537, 23363, 19898, 5318, 
	    363, 13120, 24402, 14769, 826, 10295, 13037, 8105, 12880, 18133, 14906, 11126, 
	    6717, 15, 11595, 8253, 19693, 14608, 11232, 106, 17472, 22881, 16877, 22508, 
	    21303, 7780, 8729, 16484, 14012, 9497, 11651, 2369, 14350, 6223, 24235, 12960, 
	    12127, 8344, 10427, 7378, 23457, 23188, 7469, 20313, 6956, 4962, 7332, 13287, 
	    3841, 10469, 5363, 1766, 8940, 772, 11344, 10334, 12930, 20778, 16566, 5096, 
	    4649, 4907, 715, 9489, 4945, 12986, 16354, 9715, 13946, 8314, 5792, 10319, 
	    16901, 16651, 2014, 23986, 5789, 24686, 2985, 6952, 13487, 13165, 7050, 4461, 
	    24385, 22285, 11270, 14639, 19988, 18994, 11510, 9038, 15403, 15986, 4696, 7982, 
	    8026, 9616, 17256, 13945, 17123, 24215, 5865, 17232, 2063, 10497, 23778, 10604, 
	    9848, 10463, 13534, 5844, 21801, 21287, 14904, 10331, 13620, 14651, 2906, 1296, 
	    8632, 23947, 23202, 21643, 23732, 16969, 6590, 10914, 18198, 4103, 22578, 4092, 
	    9006, 17401, 23196, 3160, 15966, 12615, 15667, 12183, 12507, 24564, 844, 6902, 
	    10524, 14399, 7511, 4758, 3476, 13275, 22561, 13464, 24790, 14555, 5492, 10198, 
	    6452, 11550, 21019, 12246, 3893, 14372, 18465, 2813, 22787, 1341, 3598, 4671, 
	    11098, 1882, 16822, 21111, 1664, 15242, 23999, 3262, 14721, 3615, 3773, 4965, 
	    22375, 14420, 19973, 11214, 5404, 9714, 24632, 13891, 5604, 9396, 7024, 4959, 
	    11149, 17158, 10947, 14732, 8480, 9573, 4144, 20260, 15081, 5619, 24546, 20635, 
	    16323, 17392, 3564, 6868, 541, 24220, 21701, 23125, 18399, 14837, 20660, 13405, 
	    6548, 21720, 23616, 5823, 5926, 4838, 15428, 3934, 20747, 22282, 13449, 11748, 
	    14576, 12299, 7612, 10363, 6635, 22984, 667, 21031, 576, 5601, 7976, 20565, 
	    7603, 11881, 9982, 13046, 5283, 23752, 16747, 19445, 19481, 11363, 16982, 18446, 
	    9105, 21636, 12954, 1202, 15261, 169, 5308, 3204, 2996, 7535, 3663, 10699, 
	    21020, 5191, 5603, 24496, 12645, 19289, 8469, 15760, 870, 22842, 15463, 455, 
	    11241, 14527, 1875, 4252, 23983, 2683, 14048, 24652, 93, 12725, 11102, 7331, 
	    10338, 10813, 22463, 16455, 11725, 17574, 12776, 10884, 14212, 8627, 12889, 
	    5631, 1859, 4653, 5133, 17501, 8993, 15073, 7922, 12753, 1632, 12544, 777, 
	    23484, 1293, 7479, 15730, 22500, 1551, 20782, 14361, 20819, 17801, 10552, 18951, 
	    22429, 11613, 7145, 154, 1113, 20738, 2331, 6705, 2640, 3815, 14587, 1075, 
	    20048, 11817, 18014, 5380, 9268, 24325, 7586, 1072, 808, 20877, 1917, 21848, 
	    24039, 24572, 6709, 6278, 9413, 9920, 11103, 20315, 932, 6944, 24262, 4347, 
	    8805, 4244, 18804, 1468, 573, 1669, 10158, 18170, 8566, 1231, 2768, 12752, 
	    24356, 9116, 2847, 16621, 24082, 8820, 16692, 88, 8725, 1430, 24901, 18566, 
	    11716, 7811, 17596, 177, 16965, 4230, 10208, 8014, 22226, 16130, 8979, 24190, 
	    43, 18612, 6571, 21481, 19616, 8071, 14670, 3278, 15194, 1356, 15027, 5982, 
	    18918, 18757, 15221, 7430, 18454, 11467, 7401, 15044, 1686, 7995, 9041, 19418, 
	    1368, 21025, 23444, 12916, 23840, 21723, 8570, 2335, 22665, 14553, 2337, 1301, 
	    14087, 9169, 17740, 3058, 22002, 16356, 22800, 8508, 6052, 22870, 20826, 3604, 
	    21804, 14195, 15090, 7756, 12348, 3051, 2117, 22191, 9284, 6073, 795, 18442, 
	    2532, 22174, 24501, 23379, 1626, 21713, 1399, 6036, 14636, 21509, 3484, 11441, 
	    3453, 13643, 13393, 18945, 17771, 10684, 1380, 13511, 15453, 17653, 14862, 8160, 
	    17997, 11884, 11991, 21189, 3129, 11980, 22647, 3622, 21873, 15191, 24772, 
	    24183, 1658, 7288, 19891, 16925, 20368, 17623, 5949, 24056, 3285, 5699, 21205, 
	    7584, 5775, 18035, 22058, 16497, 7810, 18574, 20211, 5416, 1076, 3344, 20686, 
	    1576, 11216, 23467, 1189, 24125, 19379, 14160, 19331, 13435, 21452, 19332, 8903, 
	    24789, 5038, 21493, 641, 17413, 18797, 13647, 205, 15047, 7549, 18609, 1542, 
	    2093, 20519, 15022, 18522, 14011, 13360, 22183, 16570, 11465, 8076, 23004, 1545, 
	    16950, 18283, 23405, 1849, 17296, 13565, 2024, 2928, 9913, 21897, 20244, 5551, 
	    20504, 19468, 19492, 23117, 3215, 1403, 2099, 24219, 14178, 13499, 13869, 8601, 
	    24375, 3078, 20191, 20703, 18629, 12745, 12537, 19062, 8102, 19811, 16432, 8556, 
	    18368, 10920, 24845, 24271, 479, 11773, 22795, 6544, 803, 23431, 22201, 20292, 
	    6304, 18199, 24386, 22540, 21979, 1307, 15887, 10569, 14084, 22645, 580, 18444, 
	    19385, 17929, 9096, 11745, 13252, 24774, 6579, 14716, 4274, 23773, 16793, 16755, 
	    19229, 7704, 4356, 15815, 9293, 10591, 6597, 23081, 2528, 6581, 21351, 19145, 
	    23884, 24308, 11240, 13760, 15529, 18899, 23842, 11318, 14436, 8723, 568, 17319, 
	    10287, 21219, 22382, 17403, 4582, 19467, 11666, 19346, 3084, 3469, 3939, 5695, 
	    13546, 15174, 23408, 170, 8963, 21261, 11628, 11416, 10478, 12761, 22818, 23002, 
	    21299, 19793, 12004, 20429, 10781, 19237, 1302, 18394, 15281, 22202, 16853, 
	    16392, 8405, 19651, 20491, 20739, 20183, 1812, 23870, 20360, 17611, 7870, 20756, 
	    4381, 15539, 10762, 20859, 21772, 17129, 1258, 1929, 7986, 15556, 6161, 596, 
	    9094, 6323, 24324, 2586, 22667, 21180, 9, 9970, 23691, 8850, 23599, 4588, 6465, 
	    24598, 16796, 6190, 14411, 10179, 5655, 5122, 3890, 19365, 4487, 12143, 2677, 
	    19701, 11158, 23568, 4470, 10298, 22958, 19760, 8507, 19886, 6776, 17245, 18704, 
	    10248, 23341, 15753, 2584, 22952, 15490, 16326, 1995, 14825, 1203, 7807, 12697, 
	    20953, 12237, 21181, 8687, 203, 14423, 16919, 22376, 17767, 20214, 13745, 2591, 
	    23313, 4555, 7166, 23005, 10822, 23235, 14649, 7675, 3324, 9184, 6266, 1398, 
	    14833, 2157, 24876, 9144, 22852, 19098, 15122, 22586, 23519, 16983, 8861, 6453, 
	    2500, 1295, 5039, 8361, 22726, 1909, 23559, 1691, 8487, 6843, 17485, 2095, 4704, 
	    7578, 758, 8112, 15685, 401, 23510, 22449, 7823, 18087, 20096, 21516, 16624, 
	    7462, 798, 14799, 15005, 11138, 18678, 21424, 18947, 17869, 22684, 4108, 20184, 
	    12349, 23411, 19349, 20055, 10129, 21403, 12389, 1922, 19045, 13811, 4857, 
	    12914, 5870, 10572, 10842, 18682, 1864, 11682, 4860, 5417, 12170, 3700, 20796, 
	    14785, 10488, 17970, 2884, 4871, 11626, 10096, 4334, 23724, 10166, 15410, 23494, 
	    5162, 24421, 19031, 1528, 14074, 1585, 1623, 21456, 5306, 17500, 24954, 12919, 
	    17034, 17690, 7665, 15802, 10040, 18258, 22308, 23963, 18557, 19095, 19762, 
	    15130, 16029, 9057, 20380, 18775, 11417, 19373, 1591, 2478, 13185, 8402, 22784, 
	    8233, 23238, 13594, 7149, 11509, 323, 19648, 21053, 16740, 22748, 10795, 1533, 
	    13091, 20127, 24351, 9456, 18275, 21944, 16502, 1401, 16176, 525, 16428, 215, 
	    1970, 11649, 5747, 19167, 20076, 6484, 12357, 14604, 4523, 14538, 5552, 10327, 
	    20764, 11426, 588, 3430, 2767, 15231, 16337, 18801, 22865, 13633, 2682, 13790, 
	    2169, 15112, 20763, 18660, 13948, 24080, 6788, 8265, 4513, 157, 14829, 1631, 
	    18273, 5507, 20813, 12824, 15839, 7035, 8044, 21777, 17342, 15845, 12509, 21480, 
	    17752, 21384, 9583, 6683, 12003, 794, 2048, 18443, 5106, 19714, 22152, 24086, 
	    1034, 5829, 5512, 3143, 22001, 10851, 22522, 7969, 24890, 7768, 19978, 14933, 
	    3807, 4747, 17164, 19998, 13679, 14079, 3295, 2614, 9481, 14581, 22895, 14929, 
	    14693, 24526, 9118, 2732, 16449, 20402, 15661, 14700, 1100, 4224, 16748, 22485, 
	    20047, 23469, 19516, 7184, 24576, 12227, 24465, 264, 11577, 9658, 21168, 6055, 
	    17435, 16670, 16608, 9484, 659, 22539, 16787, 2437, 5508, 13912, 375, 18856, 
	    7242, 1463, 22652, 20685, 4260, 16430, 16300, 3656, 14565, 13593, 13188, 6389, 
	    7137, 9830, 2219, 10169, 6782, 17469, 12354, 1395, 3867, 1471, 23069, 5624, 
	    3699, 19441, 12281, 12617, 12114, 22128, 4663, 14618, 1017, 3780, 8708, 21831, 
	    23971, 14200, 9279, 24180, 15045, 8789, 3103, 21640, 23938, 1262, 24051, 5430, 
	    2757, 22460, 10992, 345, 24857, 6065, 19281, 13680, 11306, 24117, 14459, 24968, 
	    5328, 8885, 1595, 2633, 17264, 8127, 15534, 8826, 18810, 2290, 15311, 22851, 
	    16613, 7963, 21588, 19671, 1647, 10647, 11153, 22740, 11668, 23459, 4243, 19519, 
	    14130, 24732, 18098, 11907, 19823, 8198, 17695, 14757, 3579, 18434, 7190, 13261, 
	    8068, 22147, 584, 18057, 21774, 377, 16402, 16373, 5077, 17872, 11183, 16329, 
	    1924, 23988, 7244, 21597, 21506, 11249, 19832, 5676, 19977, 439, 17743, 353, 
	    10644, 16825, 22734, 22100, 16406, 14419, 21504, 8349, 3942, 22283, 22049, 3021, 
	    1088, 16672, 10766, 11062, 20898, 22151, 3869, 23865, 13123, 13209, 3887, 12100, 
	    18892, 22930, 5633, 22797, 770, 21272, 9178, 20521, 13915, 22754, 15875, 23276, 
	    4346, 8811, 20641, 12944, 20822, 20117, 10814, 3394, 3830, 10949, 23105, 17556, 
	    23184, 15249, 23214, 22644, 6733, 20517, 5310, 14182, 4868, 16359, 9246, 24997, 
	    12813, 19235, 2223, 13379, 7816, 11466, 18171, 8516, 23839, 14826, 22860, 3999, 
	    11037, 16216, 19455, 5436, 24605, 10846, 20019, 14097, 22209, 15433, 14736, 
	    12758, 18107, 13551, 12859, 4902, 21762, 22660, 4418, 4216, 7348, 776, 13581, 
	    3101, 11144, 17661, 11847, 18750, 13075, 20064, 21434, 17084, 20200, 7502, 9277, 
	    16302, 8838, 23331, 22808, 2993, 23110, 16087, 20767, 20975, 6109, 14303, 18421, 
	    75, 15304, 23062, 13115, 6339, 17534, 2133, 4437, 19171, 17913, 23590, 11859, 
	    661, 18479, 12089, 5143, 24512, 22070, 23285, 10549, 2097, 5433, 5190, 21613, 
	    24542, 17679, 20640, 2775, 14161, 21176, 22779, 21213, 13368, 5375, 18726, 2480, 
	    19383, 18436, 16749, 12766, 18819, 18384, 2765, 24827, 13715, 11766, 24132, 
	    23821, 2067, 6359, 17985, 21584, 11431, 2143, 3889, 24765, 18207, 14849, 8203, 
	    15422, 1298, 18083, 19746, 24439, 21592, 7214, 10805, 6696, 15214, 5160, 19309, 
	    9221, 3257, 14298, 20663, 3878, 15745, 22815, 21575, 24957, 16334, 22161, 16063, 
	    20278, 6629, 6416, 6017, 244, 4233, 5217, 12795, 21886, 3534, 787, 3567, 7306, 
	    8185, 13855, 4011, 1747, 12674, 8961, 10440, 21994, 4682, 12744, 2947, 10232, 
	    9490, 14760, 17257, 13757, 4454, 1784, 20044, 16468, 3270, 18790, 23739, 13190, 
	    23976, 15270, 19618, 15826, 18842, 10187, 2791, 4885, 7447, 2026, 10306, 3563, 
	    1320, 16446, 22630, 20393, 13203, 7849, 1315, 24724, 24781, 14796, 4841, 10296, 
	    19374, 3791, 21201, 13151, 24643, 159, 6019, 2599, 15775, 21677, 5199, 7664, 
	    14958, 15063, 3389, 9880, 6075, 20549, 15925, 70, 10575, 19175, 13236, 1182, 
	    556, 13285, 17335, 14524, 12562, 4337, 11395, 9541, 22122, 2037, 14920, 1459, 
	    21334, 23443, 782, 8339, 5123, 13675, 2804, 10361, 280, 8261, 16960, 4778, 
	    12524, 16719, 11702, 1240, 17649, 18231, 1028, 9770, 520, 16769, 18015, 9679, 
	    11854, 10480, 3088, 17297, 19803, 7314, 12223, 20322, 17884, 12985, 20325, 
	    11255, 4802, 4604, 5165, 10775, 5045, 10731, 3725, 3779, 2522, 22217, 16636, 
	    13606, 7994, 18518, 13544, 11842, 2161, 18510, 6906, 3330, 119, 24581, 3117, 
	    10246, 3433, 23503, 364, 10435, 8929, 21606, 5984, 1014, 4981, 5809, 5287, 
	    16494, 6556, 19757, 9108, 6143, 20798, 8430, 19469, 410, 17152, 22008, 7075, 
	    22626, 14870, 22408, 17017, 3086, 11792, 23681, 127, 18284, 14472, 22637, 4317, 
	    5511, 14801, 930, 14434, 21293, 6963, 8466, 15808, 4920, 12236, 16675, 17006, 
	    2744, 24583, 14370, 6430, 21425, 23472, 3424, 10529, 17269, 23908, 5777, 22603, 
	    21443, 7085, 9525, 5109, 23362, 20106, 18672, 1335, 22479, 1291, 4178, 9077, 
	    11405, 23357, 16799, 5805, 19075, 23378, 5356, 12928, 20874, 15833, 7573, 2882, 
	    20425, 9181, 3321, 7230, 9302, 10466, 10483, 16767, 10918, 20327, 16258, 633, 
	    5053, 1720, 8777, 12825, 11106, 15425, 3192, 11683, 4402, 19694, 6108, 10452, 
	    11755, 22032, 8138, 23403, 8291, 21818, 24608, 7206, 15855, 19981, 14963, 24537, 
	    16592, 13029, 21221, 4656, 20676, 24353, 24600, 3910, 15866, 19435, 2880, 16331, 
	    10360, 11855, 5671, 11673, 16315, 2279, 4124, 24519, 18212, 23707, 977, 4566, 
	    11849, 13850, 9521, 14026, 10925, 14856, 11259, 6348, 21937, 1834, 8595, 24635, 
	    20353, 12684, 16152, 23710, 8906, 9462, 21893, 7846, 16962, 24835, 14076, 22629, 
	    22165, 19500, 19825, 12867, 16986, 8227, 1613, 13985, 21905, 7002, 17609, 4615, 
	    17056, 16538, 9862, 561, 20916, 16387, 2493, 6022, 717, 2145, 10587, 23643, 
	    1906, 8288, 15917, 10242, 407, 9779, 2624, 22721, 11832, 5983, 22366, 5764, 
	    8247, 23007, 17417, 4861, 10929, 4979, 16127, 17133, 4707, 5989, 12436, 10627, 
	    16874, 16500, 9577, 19, 22488, 16197, 2858, 484, 17498, 3586, 24206, 3545, 9677, 
	    16614, 15600, 17921, 15588, 9079, 3412, 8111, 12876, 1038, 3237, 4114, 17573, 
	    23106, 10405, 9010, 11925, 21315, 18365, 16240, 18377, 5259, 10756, 14041, 5145, 
	    1633, 4158, 17477, 16278, 15558, 22527, 14480, 23596, 3702, 11245, 8048, 5864, 
	    21910, 18523, 5546, 24976, 220, 8551, 12626, 6540, 18901, 7361, 23427, 175, 
	    3481, 13306, 1654, 20531, 12368, 15424, 3821, 10729, 16988, 10787, 23082, 16930, 
	    15939, 9883, 7148, 9781, 24484, 18418, 18500, 650, 19110, 7866, 7196, 18744, 
	    2031, 10431, 4738, 17499, 9904, 6357, 18817, 6217, 7636, 23743, 21412, 18391, 
	    2994, 16352, 17380, 21773, 9269, 7857, 15146, 5480, 6226, 14532, 5972, 8571, 
	    346, 24428, 17357, 7793, 6419, 6878, 5718, 22367, 9329, 18639, 19635, 21914, 
	    8599, 1093, 1281, 16900, 9215, 9099, 20242, 17356, 18974, 13770, 4132, 6632, 
	    22690, 3682, 20443, 24551, 19369, 23893, 13699, 4732, 21929, 19241, 10551, 
	    21442, 15488, 10472, 21512, 15850, 6444, 6605, 1682, 4329, 9956, 10336, 672, 
	    21784, 1294, 13399, 22844, 5509, 11880, 1801, 11772, 6306, 9312, 5861, 7000, 
	    5364, 18109, 16891, 7472, 3282, 22173, 24225, 7251, 1005, 5932, 9188, 22962, 
	    14384, 4998, 16138, 20976, 8475, 3667, 10352, 18676, 14910, 16416, 9638, 4829, 
	    9326, 1804, 16734, 2569, 9280, 1052, 24516, 7043, 22109, 16292, 20964, 15007, 
	    19778, 1279, 10767, 20599, 14912, 11315, 15121, 15927, 19963, 16412, 5832, 
	    13096, 4710, 22980, 16011, 11423, 21999, 6689, 1445, 5900, 20870, 15596, 1997, 
	    24378, 4200, 14683, 13531, 12364, 188, 20665, 19259, 5725, 18104, 7527, 9681, 
	    22111, 1795, 21081, 21194, 8369, 6424, 24755, 18181, 17797, 10151, 10544, 6059, 
	    9787, 17770, 16171, 1538, 15579, 3592, 9131, 6903, 11181, 24371, 23222, 6617, 
	    9518, 5220, 13192, 16222, 10234, 5379, 15677, 4848, 10023, 22891, 23009, 6502, 
	    4715, 10241, 11089, 7390, 19458, 4726, 16554, 14124, 18285, 8332, 3046, 20254, 
	    7564, 16662, 7366, 15658, 19407, 23083, 15832, 17697, 13571, 4232, 7386, 3459, 
	    5075, 15571, 24120, 13498, 1111, 963, 12326, 23734, 23578, 308, 23837, 8216, 
	    15754, 13983, 10339, 22330, 8348, 3835, 23960, 12113, 15091, 139, 8865, 23627, 
	    11184, 8537, 11752, 24006, 24152, 16145, 11972, 1523, 14158, 2597, 2819, 10802, 
	    21921, 19806, 13577, 19139, 20603, 14803, 7803, 7833, 18295, 13231, 24240, 
	    24098, 15266, 5134, 10726, 16581, 1952, 6319, 23556, 20995, 9732, 4181, 15137, 
	    5060, 6072, 2716, 10412, 889, 19215, 13388, 18376, 15195, 18024, 4987, 2313, 
	    24328, 17540, 3, 24487, 19645, 21271, 20759, 23173, 13183, 2033, 218, 10560, 
	    9928, 14544, 5559, 10326, 16496, 11767, 21526, 12380, 22194, 7031, 5886, 4603, 
	    11361, 14320, 13843, 7834, 17825, 691, 6489, 20677, 1425, 19283, 3217, 16517, 
	    14616, 7954, 1134, 149, 6817, 15166, 13932, 13925, 6921, 24184, 10171, 4211, 
	    12750, 20853, 10631, 5716, 19856, 13558, 20784, 6329, 4379, 2776, 6515, 17156, 
	    17171, 14204, 1214, 8698, 16259, 2381, 9545, 6697, 562, 1373, 6947, 23015, 
	    21190, 4719, 13255, 3092, 19170, 5589, 7163, 19073, 2283, 10098, 10858, 6254, 
	    14966, 4060, 7098, 11141, 8147, 815, 12086, 18373, 24420, 7300, 11715, 14322, 
	    15284, 11378, 19896, 18524, 24108, 5035, 19729, 2491, 3864, 4143, 285, 7392, 
	    2463, 24836, 15500, 9232, 18002, 2827, 14234, 16514, 8600, 15525, 8170, 6082, 
	    592, 5967, 14462, 16599, 19446, 17018, 12232, 18882, 17201, 5032, 15874, 544, 
	    3916, 19180, 16819, 5678, 22562, 23172, 16656, 15013, 24089, 2241, 12987, 7999, 
	    12498, 92, 23109, 1491, 4167, 3235, 8151, 10175, 5611, 21346, 13403, 22554, 
	    3296, 14256, 24070, 2417, 18363, 12793, 4373, 7510, 2170, 8425, 3466, 7229, 
	    10430, 16993, 23906, 2346, 10468, 22533, 21540, 20335, 20036, 10649, 19985, 
	    13080, 6300, 18548, 12531, 18537, 15933, 5228, 23581, 11342, 9434, 1515, 7484, 
	    8832, 15339, 15738, 21066, 10651, 10913, 15827, 18657, 1371, 8603, 6192, 3537, 
	    6996, 12384, 14787, 10418, 16008, 12433, 2705, 21744, 9625, 3986, 14612, 19077, 
	    11680, 19348, 19202, 14678, 15672, 9449, 1690, 23814, 13937, 23136, 1915, 20765, 
	    19191, 4872, 16142, 7418, 22798, 11231, 8559, 13826, 11862, 11905, 3161, 193, 
	    6168, 15578, 3795, 20687, 17865, 11636, 14001, 8749, 6078, 17646, 15296, 7095, 
	    7800, 7725, 5225, 18894, 20697, 7856, 16652, 17030, 3451, 3546, 16050, 22301, 
	    11969, 16750, 4879, 6374, 6758, 23127, 20145, 7504, 19840, 9535, 21800, 21986, 
	    15796, 15225, 2390, 2391, 10119, 19131, 7227, 1943, 6718, 9322, 18578, 878, 
	    23904, 3531, 13519, 5523, 178, 22257, 6293, 21672, 1729, 11699, 8272, 5350, 
	    5180, 24306, 11310, 6753, 18311, 9084, 6164, 20299, 14050, 24425, 7388, 7027, 
	    1643, 6939, 1023, 3012, 1142, 23665, 10473, 14484, 5800, 14793, 6337, 10324, 
	    18052, 15265, 1508, 19116, 12723, 15025, 7444, 15797, 13013, 3703, 24251, 16451, 
	    24257, 3523, 22885, 5074, 507, 5413, 1713, 22238, 22227, 6643, 3918, 15277, 
	    7210, 21242, 3026, 16529, 12300, 3206, 9867, 6201, 5025, 2922, 7072, 9258, 8493, 
	    12095, 9918, 22190, 5197, 24016, 6666, 21144, 14672, 2959, 18069, 9600, 3863, 
	    22068, 7355, 10753, 16090, 9808, 17391, 1261, 9746, 12491, 8858, 17662, 4255, 
	    20918, 9030, 7987, 22028, 22879, 24354, 4862, 3645, 5728, 13472, 2537, 10919, 
	    8584, 21498, 9239, 20537, 17261, 12338, 59, 1593, 6614, 3378, 7598, 10859, 
	    16398, 3785, 8786, 13847, 16093, 20, 21809, 4066, 2687, 6412, 24558, 12447, 
	    21835, 10380, 8282, 17811, 20575, 18991, 6908, 18115, 14800, 9307, 4929, 2091, 
	    1104, 6676, 12612, 18238, 8334, 21761, 15216, 13139, 23068, 4899, 18364, 2616, 
	    1710, 5558, 22093, 2258, 5849, 7869, 9093, 5388, 22882, 21059, 14600, 7026, 167, 
	    17148, 14644, 17722, 3132, 22641, 7164, 9666, 11135, 11999, 24327, 12757, 927, 
	    20843, 12528, 19824, 2494, 4399, 4208, 21966, 11731, 1386, 5274, 975, 1868, 
	    1531, 23322, 20406, 21635, 20941, 24898, 15547, 14719, 7699, 8977, 18172, 11343, 
	    1355, 4423, 251, 13642, 6250, 343, 10200, 21283, 10190, 16194, 1825, 11976, 
	    4312, 2734, 14175, 2257, 9733, 19359, 16852, 23417, 6715, 23487, 12724, 2558, 
	    23591, 17753, 11911, 4583, 22110, 15632, 19172, 19209, 11414, 21871, 16403, 
	    6172, 10741, 2056, 9560, 22785, 10827, 3851, 20188, 16173, 17189, 13670, 17397, 
	    2713, 453, 6262, 5338, 21948, 14898, 16427, 23968, 16823, 18927, 8798, 11404, 
	    3584, 6213, 11193, 18178, 22951, 16411, 21161, 4960, 3763, 10825, 23992, 17179, 
	    3677, 14224, 19428, 24993, 4026, 8894, 12359, 9959, 22718, 6077, 11011, 3121, 
	    1306, 666, 2710, 2572, 15758, 19413, 11238, 17652, 4769, 21197, 15447, 15801, 
	    19615, 3182, 5908, 6491, 9947, 4630, 24061, 15885, 17631, 1283, 11789, 17956, 
	    8560, 4152, 7821, 18350, 4643, 24415, 19105, 22120, 6784, 4633, 8742, 14605, 
	    13898, 3603, 24911, 4708, 16366, 12275, 11056, 18401, 13792, 20258, 23822, 
	    20468, 11261, 5478, 12406, 20009, 14211, 4319, 22056, 12386, 6855, 9698, 17854, 
	    5778, 9927, 22604, 168, 20881, 15798, 9153, 10451, 5660, 23220, 9479, 1070, 
	    19595, 18973, 11877, 24966, 11194, 24198, 23204, 10743, 20417, 18828, 17517, 
	    11489, 19668, 7854, 24714, 15367, 18277, 10985, 17488, 10789, 21974, 9241, 
	    17999, 23560, 21062, 11279, 7791, 1141, 9955, 6442, 13771, 15338, 22326, 2481, 
	    2514, 3696, 18325, 22400, 21697, 21092, 15618, 13791, 7123, 10970, 14631, 18685, 
	    16821, 15096, 11476, 24426, 6002, 8509, 15614, 23178, 12789, 14784, 24036, 
	    17147, 16801, 15503, 16743, 24127, 209, 8303, 11296, 16849, 21166, 23372, 14811, 
	    11075, 3041, 15189, 22789, 10713, 13646, 10354, 1002, 5592, 21805, 8667, 19306, 
	    8483, 12257, 21402, 21087, 227, 6159, 20583, 11576, 7841, 3263, 5192, 19060, 
	    1252, 4020, 1779, 11685, 19796, 9661, 483, 10186, 8266, 3937, 4577, 21003, 3746, 
	    17918, 14684, 1157, 5952, 9119, 6222, 22921, 22655, 2780, 6125, 10484, 11757, 
	    7528, 9929, 1393, 9550, 493, 20886, 23516, 13958, 16577, 2649, 2232, 17719, 
	    10072, 8354, 22216, 17608, 24063, 10532, 2187, 9135, 458, 21764, 23256, 9814, 
	    24311, 23037, 7473, 11025, 18950, 437, 18039, 16546, 8219, 3524, 9104, 14299, 
	    12781, 12192, 17024, 430, 9034, 21559, 6360, 21549, 3032, 12828, 5124, 21348, 
	    9244, 3358, 4556, 10899, 12895, 24467, 16423, 18463, 23147, 16712, 6482, 20923, 
	    5947, 3415, 4426, 18152, 13372, 6106, 17861, 17186, 2321, 16112, 11034, 30, 
	    6803, 1693, 14210, 12580, 2592, 3252, 6607, 22948, 24531, 19117, 12870, 16631, 
	    12482, 3405, 1287, 21426, 2571, 11743, 14363, 2548, 5744, 21125, 16252, 18741, 
	    12516, 22222, 8333, 18154, 21084, 20855, 16271, 17840, 24947, 8888, 21155, 
	    10453, 24233, 15327, 22402, 19587, 3435, 15615, 4792, 13923, 10003, 20093, 
	    10885, 23374, 17749, 4500, 20270, 18561, 23026, 12175, 24962, 12712, 804, 21154, 
	    23649, 7046, 19276, 14697, 2487, 5026, 12445, 20413, 4570, 18242, 22695, 7640, 
	    21216, 15609, 7150, 19931, 18003, 19295, 1817, 12794, 24693, 20652, 24090, 
	    21255, 23977, 20263, 20301, 12502, 20251, 23415, 1768, 18634, 8166, 13204, 
	    23524, 440, 925, 5735, 21684, 17698, 9586, 19841, 8223, 8783, 11157, 24027, 
	    7047, 242, 12736, 2318, 9685, 3949, 15201, 7063, 20065, 9078, 307, 17650, 15621, 
	    18513, 2566, 23425, 18843, 22868, 6984, 2523, 2860, 10213, 17966, 4102, 18707, 
	    13711, 11819, 1201, 3919, 21320, 5922, 15742, 7991, 4043, 16572, 3572, 5072, 
	    24310, 13992, 22298, 5097, 9409, 18100, 2841, 15941, 21110, 16722, 20026, 14798, 
	    24654, 15485, 24323, 19904, 9580, 20958, 21083, 8194, 13146, 13444, 9278, 13178, 
	    5399, 23712, 18237, 19152, 11535, 15483, 6279, 7089, 9264, 3111, 17240, 13162, 
	    4298, 18331, 11128, 15060, 24289, 1352, 14603, 12416, 18699, 2003, 7567, 23879, 
	    24648, 19183, 21520, 16283, 7654, 21726, 21591, 16047, 22750, 21576, 23252, 
	    8401, 2601, 21932, 293, 5520, 12344, 14815, 13603, 3822, 2520, 24721, 13979, 
	    18597, 1057, 20533, 12, 11055, 24479, 11875, 4702, 12743, 7913, 20133, 11517, 
	    4258, 24644, 3422, 5936, 1611, 7574, 19204, 754, 17882, 14908, 6602, 21396, 
	    14189, 470, 12297, 6286, 19966, 16836, 15061, 19556, 9230, 2853, 12686, 10252, 
	    354, 68, 8079, 2685, 1848, 23014, 20542, 18606, 3628, 8224, 5182, 4973, 1912, 
	    23221, 12181, 3747, 6406, 11321, 15233, 21478, 3332, 13214, 13043, 9086, 24969, 
	    17010, 24840, 9999, 5488, 24768, 698, 12369, 4728, 11592, 19188, 23302, 13256, 
	    22802, 2637, 13281, 11411, 11251, 5395, 11955, 1714, 7916, 5588, 7690, 9893, 
	    12692, 16454, 17604, 17439, 10797, 20567, 22470, 3936, 15721, 7769, 22242, 
	    13023, 3098, 6025, 14573, 18847, 3326, 13751, 13664, 12569, 2336, 4145, 8153, 
	    23525, 4714, 749, 21140, 7412, 6827, 11865, 24806, 9393, 20376, 14959, 1881, 
	    13944, 18735, 6198, 10671, 1786, 23886, 17836, 22550, 5050, 15113, 24859, 17622, 
	    12835, 3048, 21331, 16775, 23937, 21519, 8530, 21212, 8801, 3413, 14441, 11776, 
	    1935, 8645, 2840, 7059, 5323, 22661, 20962, 18771, 17978, 3077, 19181, 15052, 
	    1857, 18347, 15607, 14867, 12905, 5545, 10567, 18891, 3842, 2856, 7947, 11591, 
	    8505, 9648, 6990, 12713, 20480, 11071, 3897, 5066, 18694, 934, 24692, 15416, 
	    20927, 5647, 11872, 4296, 11840, 23464, 24547, 18343, 5874, 19644, 117, 18701, 
	    2256, 2294, 19718, 10041, 4135, 20579, 17944, 15520, 18705, 22941, 21528, 1191, 
	    15951, 13805, 20453, 13355, 713, 20973, 12022, 17302, 2089, 642, 12270, 23661, 
	    7066, 15554, 12387, 23786, 1798, 19851, 9596, 5567, 24604, 18281, 23149, 20241, 
	    4225, 24955, 12125, 2737, 2, 20282, 5214, 10603, 19036, 14703, 19508, 13133, 
	    11008, 21280, 7647, 22585, 434, 3819, 7231, 4358, 9617, 18888, 15569, 4357, 
	    17506, 9228, 11307, 15399, 17816, 14928, 16297, 4791, 7083, 14414, 10662, 3557, 
	    8228, 9133, 7862, 11428, 10345, 2370, 3220, 1571, 11580, 13170, 21563, 10962, 
	    14706, 23162, 18293, 14036, 7193, 19184, 11536, 21749, 1627, 6804, 20326, 9147, 
	    6435, 13353, 6392, 5472, 22453, 5561, 5706, 22024, 11944, 12934, 3718, 16186, 
	    13673, 9798, 24522, 13636, 13495, 22697, 6244, 7202, 21872, 22135, 18498, 9366, 
	    14860, 14049, 9463, 2948, 16263, 4390, 12682, 11480, 21444, 18004, 22552, 14466, 
	    4930, 3201, 16872, 6603, 2576, 4820, 20178, 21538, 12792, 15772, 21736, 22906, 
	    17449, 472, 19561, 15303, 6133, 19457, 19512, 7104, 18417, 20538, 16394, 23247, 
	    11899, 3633, 5469, 2282, 1727, 15167, 23022, 4228, 24384, 3955, 23039, 4191, 
	    14000, 22663, 24320, 24807, 19804, 22973, 4205, 15824, 9965, 22084, 3397, 20125, 
	    10174, 22832, 4761, 10180, 10074, 6042, 952, 2068, 22886, 18940, 1248, 5883, 
	    9036, 17277, 24491, 14895, 6740, 18607, 16201, 18683, 8837, 24791, 468, 18514, 
	    17824, 5269, 563, 8824, 14955, 24193, 5382, 16462, 22300, 20252, 24578, 12552, 
	    6135, 11235, 15593, 15931, 14595, 304, 20484, 18623, 22698, 24173, 4605, 17998, 
	    5934, 21231, 22335, 11051, 3097, 3931, 21593, 12959, 8756, 24300, 18136, 8892, 
	    10652, 13830, 94, 23810, 4502, 8342, 9973, 4173, 18812, 6156, 21193, 19353, 
	    9157, 11947, 10696, 17988, 13414, 14710, 14944, 18582, 4056, 6771, 15573, 8006, 
	    24106, 4753, 15612, 1222, 5400, 22878, 9391, 22105, 10778, 3407, 18029, 5729, 
	    17663, 14444, 14107, 6687, 10062, 18556, 4406, 21964, 24647, 20321, 19736, 
	    16568, 755, 14037, 202, 20625, 1131, 24992, 3047, 3054, 8625, 2679, 11276, 9680, 
	    1049, 19821, 22392, 21544, 10664, 5714, 9958, 13530, 23982, 5953, 1572, 970, 
	    7324, 8047, 6058, 6885, 22210, 19488, 20312, 5226, 19509, 12163, 24886, 6924, 
	    16601, 14882, 5494, 14845, 12254, 4119, 7817, 5723, 17022, 23856, 13592, 12879, 
	    1757, 2704, 21699, 12329, 22184, 226, 19865, 12164, 17911, 17714, 15900, 11147, 
	    6267, 19914, 898, 614, 8617, 7389, 18028, 3428, 15638, 16567, 17747, 12705, 
	    8704, 6097, 1559, 763, 12427, 16248, 5986, 6698, 16298, 18390, 6905, 3600, 
	    17165, 16126, 6093, 20416, 6742, 4186, 18774, 3480, 11242, 15475, 9390, 20552, 
	    15910, 24246, 13879, 23171, 871, 4889, 10116, 22134, 16307, 5527, 13125, 4383, 
	    8221, 11878, 15097, 2940, 17194, 18702, 14960, 18046, 17973, 18, 14578, 8944, 
	    4954, 13056, 11647, 12105, 13206, 4675, 24318, 6958, 23420, 6550, 20271, 6796, 
	    13407, 12840, 13131, 18989, 223, 20618, 19429, 928, 9571, 6134, 3479, 1728, 
	    15405, 1486, 4986, 23113, 4581, 21369, 6272, 4464, 7427, 6139, 855, 13868, 
	    20590, 19619, 8558, 4955, 15499, 245, 11301, 12370, 14170, 13363, 18645, 11778, 
	    9408, 5801, 5817, 8263, 15317, 10419, 20189, 24474, 20987, 18724, 6119, 1914, 
	    22546, 17523, 20679, 19948, 12513, 18187, 452, 10102, 10144, 1872, 21612, 8967, 
	    21689, 11273, 16157, 1325, 19798, 12301, 459, 19434, 18841, 13142, 9750, 13701, 
	    16028, 2521, 9235, 21622, 1783, 12026, 7662, 2330, 1246, 806, 22918, 6703, 
	    10975, 8312, 23797, 7716, 14073, 17845, 11485, 6915, 1211, 18298, 14469, 9400, 
	    22518, 14226, 15213, 24009, 18718, 6514, 24161, 17673, 14475, 2774, 23265, 4740, 
	    13479, 19260, 6993, 8351, 201, 3365, 6694, 10533, 17076, 15203, 14967, 19637, 
	    14019, 10020, 18085, 2662, 9870, 10512, 10089, 17864, 4576, 14723, 22121, 9713, 
	    60, 15154, 3790, 2577, 10069, 5233, 3139, 17476, 8385, 14395, 10233, 11329, 
	    1082, 15362, 1656, 9285, 22668, 19578, 1858, 5129, 12829, 8912, 10479, 4553, 
	    18814, 22320, 6090, 6672, 13398, 21717, 5396, 12553, 12358, 16914, 736, 11672, 
	    21704, 19838, 7576, 14647, 3357, 21603, 12488, 19979, 7740, 18306, 16754, 15437, 
	    15583, 14543, 8590, 17975, 15038, 19495, 13860, 23854, 23957, 20477, 3820, 5441, 
	    14611, 17950, 4985, 21169, 11929, 17877, 12940, 14321, 6420, 13787, 6582, 9778, 
	    7836, 5155, 12540, 739, 21601, 14266, 3146, 15840, 1725, 1494, 8936, 21044, 
	    18896, 9711, 4458, 11499, 13033, 13632, 4339, 12351, 20195, 14133, 6749, 24230, 
	    9993, 578, 11468, 13838, 21406, 12431, 17241, 9042, 7305, 21751, 16132, 14828, 
	    20492, 9683, 17528, 498, 5581, 24741, 20503, 22372, 16742, 3011, 10256, 21903, 
	    20207, 18294, 333, 15388, 3134, 21813, 862, 22082, 1831, 10596, 4439, 3187, 
	    17288, 19200, 22065, 11742, 19270, 23535, 16981, 10658, 9110, 3144, 9553, 14131, 
	    20893, 23949, 19658, 14588, 7642, 15172, 2017, 19069, 24014, 22728, 21667, 
	    12373, 17965, 2035, 19326, 24842, 2602, 17271, 6745, 3287, 4975, 6798, 19905, 
	    3904, 8346, 14597, 24562, 23940, 8200, 18818, 10642, 13336, 2976, 11829, 7880, 
	    22303, 12216, 9332, 15180, 18456, 8406, 14301, 17723, 20001, 20177, 4094, 20869, 
	    4575, 3193, 21950, 3642, 668, 14570, 18007, 8575, 12075, 5383, 17412, 6004, 
	    14250, 17678, 16360, 20314, 1566, 8943, 18938, 19173, 19628, 16647, 17151, 2495, 
	    5948, 16380, 10751, 13960, 15543, 460, 4705, 24649, 21579, 109, 5901, 5785, 
	    7234, 17414, 18972, 3527, 23243, 5894, 7997, 11001, 11118, 7224, 21788, 13191, 
	    17826, 15533, 20355, 9739, 18280, 1789, 19759, 11450, 14148, 18619, 22167, 
	    22253, 20527, 19284, 3716, 17313, 21035, 18267, 21277, 8012, 11782, 951, 15603, 
	    19731, 21223, 1830, 16084, 6346, 2862, 20059, 17105, 5177, 9437, 1219, 20649, 
	    13243, 14083, 8450, 21783, 7176, 24045, 16447, 17182, 19895, 1297, 1370, 12493, 
	    4737, 17870, 15271, 13975, 21541, 4754, 13761, 22968, 19650, 9743, 3981, 12801, 
	    719, 11691, 22306, 14099, 20186, 13957, 24372, 24579, 4414, 11512, 24585, 2427, 
	    3400, 4898, 3308, 23674, 14339, 12152, 22991, 873, 7396, 16341, 8604, 23046, 
	    12251, 3273, 5210, 7971, 13269, 21101, 5724, 13456, 10508, 22299, 21124, 22415, 
	    17791, 4090, 5499, 15347, 19074, 16683, 6176, 20397, 11978, 8800, 8692, 17305, 
	    630, 20253, 3130, 18356, 20931, 5999, 8785, 2615, 6519, 24936, 20194, 23491, 
	    5103, 17209, 4259, 19025, 6593, 9198, 7853, 752, 3106, 4687, 17026, 4405, 21587, 
	    9988, 7077, 8919, 15690, 8611, 23232, 7213, 20704, 12205, 263, 1487, 7303, 
	    15432, 10311, 11587, 1164, 19366, 2442, 954, 299, 24972, 3156, 23268, 574, 
	    15756, 22790, 7524, 16703, 543, 10222, 22213, 1628, 24283, 14397, 9906, 12862, 
	    360, 17541, 17708, 14768, 15598, 4062, 12264, 20296, 504, 8386, 21349, 9031, 
	    16099, 3562, 21590, 6573, 23157, 20261, 665, 12076, 21309, 22887, 17031, 461, 
	    4163, 3635, 3804, 11262, 4698, 10837, 2308, 8192, 11360, 14292, 12031, 9296, 
	    7336, 24443, 8296, 3984, 5727, 21604, 11484, 4506, 8517, 1492, 7569, 5807, 
	    15560, 3826, 16691, 24470, 22063, 324, 21600, 9410, 8447, 5907, 9219, 15594, 
	    23000, 15484, 14140, 20284, 11928, 13528, 390, 22978, 15541, 6802, 7344, 10881, 
	    4561, 24865, 22548, 22364, 17183, 853, 19593, 22055, 9070, 12864, 1269, 12005, 
	    7757, 20740, 8710, 6349, 16888, 15259, 16869, 6935, 4294, 14376, 13580, 10368, 
	    876, 2919, 19944, 15041, 2805, 15288, 22483, 5965, 2692, 15937, 24510, 15902, 
	    24580, 18506, 3373, 22775, 3510, 15053, 11924, 16247, 2956, 12778, 6857, 1755, 
	    20014, 23771, 1271, 15648, 21302, 20116, 10149, 10928, 12318, 21068, 3630, 
	    13874, 19708, 8059, 6831, 8056, 16813, 22316, 21535, 599, 11445, 13863, 1450, 
	    16448, 18305, 24189, 23946, 17896, 10009, 13716, 743, 14346, 19251, 7908, 23249, 
	    2666, 22038, 3269, 3956, 21759, 6722, 2876, 13839, 8492, 20997, 11191, 8502, 
	    9922, 13055, 12449, 14272, 5869, 4524, 19957, 327, 2686, 9229, 23099, 2536, 
	    8062, 24144, 11662, 23409, 16241, 24573, 15342, 5842, 3501, 24020, 971, 17214, 
	    20566, 9080, 8744, 15468, 17210, 5658, 21286, 23867, 17088, 19703, 11655, 7610, 
	    22092, 22414, 11105, 23273, 15369, 7804, 12214, 3874, 21370, 15182, 10316, 
	    15781, 12785, 10877, 6649, 18340, 10031, 16512, 7514, 12302, 5273, 9662, 8898, 
	    18925, 15069, 13221, 20904, 19412, 16042, 13247, 18935, 8106, 17399, 6490, 
	    20535, 19164, 1245, 5754, 6112, 13366, 849, 16841, 11598, 17046, 6206, 4493, 
	    14352, 4091, 5623, 1096, 22526, 11325, 18137, 9992, 7966, 23995, 3167, 22239, 
	    22605, 19408, 20588, 1603, 22796, 15996, 10930, 5132, 10738, 22592, 23017, 4052, 
	    9624, 12605, 4813, 6981, 7694, 23926, 18034, 11370, 7481, 21843, 3199, 7140, 
	    24706, 23095, 21691, 24794, 24504, 9369, 8791, 12244, 2875, 11403, 1095, 15312, 
	    8110, 24213, 20302, 20536, 20077, 18264, 9061, 2893, 21779, 11629, 19873, 14355, 
	    17807, 21454, 22104, 21619, 6822, 4891, 5821, 7175, 3661, 22716, 19256, 10382, 
	    9807, 3301, 4194, 16565, 11449, 13533, 16338, 12122, 9394, 2533, 14201, 8987, 
	    2742, 5470, 4016, 20508, 2019, 9087, 6474, 2222, 1138, 7460, 5957, 11212, 16861, 
	    15929, 21666, 6554, 23035, 22814, 21195, 4565, 16223, 13523, 5346, 7400, 22160, 
	    15413, 4782, 15767, 17228, 22441, 17621, 17993, 18027, 21972, 17516, 7044, 
	    19606, 5673, 5136, 11860, 558, 735, 2316, 18920, 17432, 13, 18554, 1455, 24459, 
	    23309, 2920, 17440, 22708, 22224, 21262, 17353, 1303, 20276, 2108, 8761, 12508, 
	    9378, 10721, 13788, 17247, 12481, 11892, 11795, 2600, 13282, 16684, 22912, 5090, 
	    18912, 8580, 6362, 13878, 286, 9866, 14296, 15869, 8258, 23634, 11444, 890, 
	    10764, 23769, 4148, 14333, 23349, 10924, 19507, 3075, 16424, 20878, 8383, 10749, 
	    24222, 3943, 20647, 19808, 4771, 14724, 13794, 20779, 2983, 4430, 179, 572, 166, 
	    21220, 11618, 4112, 6583, 5431, 12922, 20741, 17699, 22352, 23509, 11335, 15181, 
	    14233, 17433, 18809, 19258, 15698, 11223, 20032, 16862, 21556, 1408, 2783, 1698, 
	    13895, 1361, 19487, 18214, 12714, 4015, 10932, 11409, 21427, 12742, 4622, 22073, 
	    1600, 12334, 10922, 7767, 6455, 9804, 20911, 5023, 3945, 7927, 21826, 8923, 
	    9934, 24049, 22791, 11720, 5737, 5896, 7287, 8444, 1990, 20757, 20049, 7950, 
	    18470, 18032, 3043, 8107, 8007, 16506, 19292, 10022, 12700, 15307, 16031, 6875, 
	    5093, 15984, 18889, 8230, 18110, 14172, 17065, 24136, 6238, 18860, 2681, 2807, 
	    3401, 17496, 20121, 15019, 20464, 3707, 9347, 12929, 4498, 15287, 12884, 23620, 
	    14274, 2002, 5406, 4000, 19976, 11791, 23142, 23277, 19048, 16470, 12381, 18467, 
	    19144, 2146, 14765, 12708, 7518, 20500, 1614, 17881, 4557, 8677, 12547, 1482, 
	    11195, 9085, 19600, 11113, 4783, 7537, 22139, 15323, 4189, 21114, 13390, 1452, 
	    13644, 5970, 531, 13964, 22807, 10954, 23585, 12118, 23107, 16482, 8621, 5688, 
	    13049, 9015, 15668, 12332, 14056, 18679, 2978, 5183, 20431, 17051, 15202, 7267, 
	    13678, 1409, 7054, 10780, 20411, 13651, 24448, 18635, 23384, 23658, 21641, 
	    11508, 8539, 15436, 24713, 14494, 16330, 915, 16627, 18268, 15613, 14474, 17675, 
	    9145, 21814, 13207, 6121, 4795, 5579, 19424, 6434, 20479, 14823, 24407, 3929, 
	    12994, 6936, 9340, 20965, 22926, 17014, 5628, 2659, 22228, 16185, 3833, 15000, 
	    11807, 9529, 20828, 1506, 9606, 6102, 11319, 16217, 19157, 389, 21961, 6604, 
	    985, 24216, 5232, 20970, 23690, 12812, 12312, 16489, 292, 16576, 15332, 23653, 
	    8842, 10835, 13994, 18205, 15446, 8132, 24769, 706, 11977, 10984, 4701, 5185, 
	    11599, 24440, 23548, 24067, 5218, 1923, 8199, 15804, 7036, 8609, 8841, 996, 
	    7594, 10636, 8294, 22186, 13026, 6575, 4268, 19845, 19846, 1767, 11348, 2464, 
	    21499, 17239, 20286, 20770, 2196, 21925, 16166, 22090, 19505, 17130, 23935, 
	    12872, 19969, 18143, 15955, 1165, 20522, 15372, 23757, 3634, 13550, 1420, 14069, 
	    6197, 21153, 11844, 23970, 5610, 17275, 8283, 5163, 14506, 20895, 6955, 20085, 
	    23881, 8268, 21079, 14854, 931, 5763, 24418, 13290, 15524, 21792, 14374, 9941, 
	    4419, 20168, 22975, 16523, 19883, 21269, 20948, 289, 4342, 17445, 12603, 13467, 
	    22138, 7018, 96, 8058, 968, 22955, 2868, 15397, 6277, 22069, 18113, 19661, 6129, 
	    523, 4308, 7142, 6271, 20896, 19706, 12418, 23185, 2655, 18169, 15087, 1336, 
	    11264, 1667, 22286, 19177, 17677, 20920, 24523, 17299, 14688, 142, 5640, 21908, 
	    19913, 19024, 21163, 20690, 24538, 9233, 24347, 13063, 5088, 17064, 13332, 
	    18037, 14119, 12284, 14504, 17526, 24388, 2242, 7850, 21437, 21319, 9700, 20694, 
	    5643, 11397, 15074, 22009, 9945, 18967, 12911, 8998, 1433, 3769, 20577, 10897, 
	    24157, 13102, 23632, 7240, 14812, 19625, 6396, 12868, 14534, 21455, 17633, 
	    12409, 1300, 11142, 11964, 2799, 12842, 8353, 19750, 18914, 10183, 1847, 15949, 
	    1382, 14008, 9881, 14064, 14251, 7709, 18825, 6451, 2475, 463, 3837, 21383, 
	    4198, 22458, 10118, 7483, 8694, 19766, 9894, 22247, 19421, 11589, 1192, 23718, 
	    16135, 24210, 8902, 16526, 302, 21074, 24331, 10250, 14899, 15904, 21106, 688, 
	    12653, 16151, 14117, 5881, 23991, 20408, 13152, 11396, 10437, 2802, 20924, 
	    14836, 4297, 8639, 15708, 14316, 16530, 8175, 12848, 18921, 3513, 9641, 1811, 
	    20324, 8855, 24481, 3472, 379, 4104, 19378, 13193, 18509, 4234, 19186, 15190, 
	    7915, 7592, 4007, 12791, 3639, 14972, 233, 6248, 5563, 19054, 6829, 23098, 1174, 
	    16371, 4836, 13753, 6417, 7131, 5391, 10796, 12067, 13317, 2975, 16097, 18621, 
	    4751, 6530, 13309, 1529, 10320, 14183, 6492, 17937, 22794, 3940, 18494, 6980, 
	    17409, 21034, 9337, 16915, 18725, 19271, 1249, 24933, 6789, 6713, 7755, 6427, 
	    4905, 9697, 7844, 448, 19354, 22256, 2815, 124, 20534, 16765, 854, 19564, 24313, 
	    15037, 3207, 13710, 13233, 1961, 4067, 20662, 11066, 8553, 16389, 10939, 3154, 
	    13110, 10143, 13616, 18709, 6098, 24495, 6205, 12028, 20775, 21517, 15145, 6532, 
	    21057, 11457, 9231, 11309, 24658, 3921, 6313, 18051, 860, 16641, 23614, 12002, 
	    15961, 5128, 11521, 24004, 10300, 11340, 21137, 6160, 11635, 9362, 3792, 18979, 
	    13890, 7779, 7448, 14229, 8986, 7034, 24021, 21186, 3119, 5525, 22696, 16190, 
	    23583, 2942, 21722, 11704, 6202, 19630, 486, 17636, 5960, 12980, 17889, 22279, 
	    21148, 15414, 12180, 9566, 15254, 9212, 4069, 5211, 17178, 11038, 17280, 16970, 
	    8764, 10494, 20459, 408, 11761, 17817, 4539, 15923, 13648, 6410, 15472, 17953, 
	    4607, 23898, 15934, 12393, 3793, 7408, 2534, 309, 23421, 17511, 23657, 16784, 
	    8952, 8985, 4315, 7631, 2773, 9363, 16051, 24663, 840, 15998, 6432, 10686, 3577, 
	    23522, 11825, 7255, 9931, 7968, 16605, 2470, 24619, 16634, 1524, 20165, 4169, 
	    1597, 2654, 9011, 17304, 866, 19735, 8218, 3883, 14593, 4755, 19401, 5963, 
	    24170, 19339, 5081, 23642, 15220, 12967, 12035, 18637, 4335, 18511, 23096, 6316, 
	    14735, 9048, 3861, 12437, 3974, 7038, 24737, 3395, 15650, 6552, 23829, 9567, 
	    16037, 10588, 4284, 1032, 9741, 24710, 8407, 3246, 10784, 382, 10578, 20742, 
	    21227, 20432, 5540, 995, 17318, 20382, 17402, 7812, 17536, 7830, 17600, 15318, 
	    13541, 19159, 2820, 11610, 8467, 893, 7822, 18278, 14135, 19299, 17586, 6596, 
	    16053, 19086, 13677, 10257, 19980, 23688, 12676, 14034, 23079, 14184, 6830, 
	    23306, 3093, 11690, 10450, 13164, 6138, 18168, 12739, 8043, 14525, 9775, 24726, 
	    18378, 992, 9499, 3382, 8449, 6071, 22677, 14335, 17001, 16875, 15820, 18936, 
	    23562, 2194, 11386, 8097, 4568, 1107, 9632, 15200, 11268, 16002, 6894, 21817, 
	    8307, 3291, 11841, 21533, 3964, 3328, 4227, 366, 14755, 5138, 4912, 6443, 2450, 
	    21822, 7221, 19422, 9417, 18869, 10793, 2735, 23432, 18360, 3095, 14695, 16573, 
	    2931, 10136, 6712, 6409, 19539, 12134, 15735, 18088, 21234, 10259, 11192, 24808, 
	    10703, 1247, 23144, 16951, 23038, 521, 18476, 3455, 11435, 12608, 8114, 1493, 
	    21158, 6459, 8991, 6794, 4652, 2972, 21435, 10585, 4730, 6256, 19566, 23742, 
	    14900, 6289, 21016, 12989, 8009, 23390, 20788, 16351, 19526, 14017, 8640, 18982, 
	    9889, 6545, 7616, 21416, 1278, 12441, 20568, 15846, 12339, 15451, 17372, 15243, 
	    17721, 8118, 3689, 13028, 6469, 11460, 18478, 10586, 9967, 8577, 15694, 12221, 
	    16467, 3676, 24850, 23790, 3849, 495, 12212, 4811, 6310, 6473, 6149, 15168, 
	    20046, 14914, 12655, 21085, 5292, 14863, 18766, 23486, 9548, 7497, 4179, 7434, 
	    11010, 4824, 10830, 3558, 3601, 18902, 13420, 8797, 19844, 4869, 19655, 705, 
	    14500, 2904, 1311, 18698, 9989, 912, 7351, 18189, 8533, 21933, 14336, 9950, 
	    18270, 6542, 19816, 5530, 8870, 11239, 9805, 18493, 22876, 10564, 20834, 18946, 
	    24729, 9186, 15313, 15198, 10645, 20514, 1885, 17962, 17489, 16928, 17561, 
	    16897, 7692, 19596, 20002, 14783, 8042, 10366, 21350, 5438, 17552, 210, 12802, 
	    12065, 12716, 21965, 20163, 18259, 21981, 21659, 18431, 709, 17174, 3840, 702, 
	    3318, 1596, 17252, 907, 4883, 18608, 8365, 11570, 20875, 5768, 10090, 18730, 
	    18433, 7673, 13239, 5831, 22979, 19213, 5454, 13896, 16374, 18185, 4952, 18410, 
	    12385, 22251, 18248, 12646, 11896, 23831, 2785, 17746, 22858, 16515, 6185, 
	    14949, 2619, 9513, 6467, 18988, 15383, 14271, 14529, 15619, 10676, 18883, 9360, 
	    23304, 18221, 13266, 11110, 2267, 4882, 6039, 8586, 7952, 15071, 7507, 12764, 
	    14838, 24437, 12471, 23132, 12777, 5871, 7902, 17422, 8829, 8500, 17243, 6957, 
	    8154, 24539, 7789, 16929, 21448, 3340, 18666, 2524, 2702, 18937, 18695, 7974, 
	    21395, 9581, 9761, 1114, 4164, 24095, 16746, 14104, 23441, 11151, 20992, 9752, 
	    587, 17979, 6557, 13970, 4162, 4969, 24688, 10708, 2787, 4942, 22956, 22725, 
	    17590, 20901, 7983, 22018, 6355, 18826, 8183, 9720, 24699, 18396, 7264, 16984, 
	    6704, 21861, 20914, 4428, 23887, 2444, 10182, 8181, 18753, 497, 7759, 7832, 
	    9227, 204, 744, 15029, 13069, 1377, 10891, 10870, 22947, 15999, 20945, 12550, 
	    11493, 7626, 10801, 18425, 4970, 24380, 18438, 6546, 4036, 14915, 14364, 12395, 
	    22601, 21954, 9327, 23262, 18079, 23352, 18755, 15252, 22959, 23762, 21557, 
	    1984, 4267, 13870, 22749, 22765, 10350, 229, 14809, 4978, 1871, 11733, 3881, 
	    19484, 24409, 4847, 18000, 6779, 4448, 3818, 10861, 8541, 6208, 23598, 20050, 
	    12388, 16931, 20810, 18091, 4826, 15417, 22520, 9100, 21485, 4213, 1947, 24393, 
	    23662, 2674, 12325, 5918, 8882, 22854, 4490, 17952, 12011, 14007, 13595, 18308, 
	    11898, 1589, 3377, 17530, 8896, 2239, 15681, 20377, 21505, 12623, 11760, 1392, 
	    13892, 15527, 23049, 6402, 962, 12675, 6440, 5484, 11083, 2955, 22317, 22744, 
	    10156, 6297, 19082, 4664, 21940, 16668, 6495, 11706, 8769, 21017, 14077, 387, 
	    6049, 2135, 1051, 21827, 22679, 2461, 9568, 17320, 8969, 16104, 12877, 7861, 
	    12139, 18813, 12956, 14042, 3643, 1268, 7653, 5179, 11384, 20136, 12598, 3490, 
	    23465, 21061, 13645, 14106, 11117, 8335, 20666, 16299, 16068, 10475, 8787, 3381, 
	    11060, 273, 8889, 4773, 19103, 10879, 1181, 13998, 12869, 9758, 2488, 17660, 
	    11867, 19125, 5172, 10557, 23553, 24454, 15140, 8739, 1979, 24031, 18939, 15737, 
	    1926, 21982, 6233, 11169, 11452, 19908, 10275, 7283, 15825, 21521, 9628, 22976, 
	    12452, 15142, 22478, 16800, 17668, 13607, 17532, 13292, 9482, 6865, 24205, 590, 
	    23861, 4322, 12201, 5402, 22223, 17924, 4781, 24770, 6558, 16313, 21806, 18314, 
	    16899, 21566, 16022, 22664, 22584, 10216, 2953, 8390, 22565, 8327, 4716, 2908, 
	    4305, 21832, 1927, 13692, 1194, 17450, 10340, 3824, 23944, 4817, 24403, 3118, 
	    21436, 17687, 22903, 10046, 3754, 11043, 6030, 20119, 7859, 5587, 19454, 21400, 
	    15890, 14470, 16250, 18491, 12309, 21058, 11529, 9023, 9629, 20111, 18570, 
	    13341, 14887, 12219, 20541, 11068, 16251, 9675, 7943, 19072, 1127, 9618, 810, 
	    8457, 19510, 12976, 257, 21151, 8297, 15586, 10623, 14334, 13815, 12407, 4599, 
	    22374, 8477, 3799, 5215, 14005, 6884, 9441, 1904, 8398, 1808, 17798, 24550, 
	    21543, 5760, 6659, 5216, 10917, 23097, 3854, 8113, 24920, 13875, 12042, 3034, 
	    18084, 12585, 10018, 17045, 2946, 3540, 4068, 10878, 6230, 15264, 12804, 24121, 
	    19250, 2339, 4723, 8515, 13969, 15415, 4579, 16908, 4984, 6675, 9436, 22492, 
	    20684, 18075, 16560, 9838, 20095, 14454, 22062, 5405, 3596, 22245, 16818, 19351, 
	    17295, 21433, 12696, 16442, 20042, 14368, 19015, 20375, 23315, 6372, 3729, 7588, 
	    12551, 16175, 17897, 13736, 5605, 10120, 23197, 721, 8726, 14704, 11676, 8827, 
	    18202, 12817, 2038, 18287, 13883, 12772, 24981, 12527, 17638, 7397, 11180, 9737, 
	    21080, 5969, 1963, 22627, 21380, 20543, 5067, 12948, 2580, 5975, 2890, 13608, 
	    12470, 4742, 9425, 9367, 14619, 8030, 16520, 7824, 18650, 5008, 8474, 13074, 
	    23307, 1994, 8210, 15550, 9154, 3530, 9601, 2103, 7813, 3664, 16019, 7029, 
	    22428, 14923, 9693, 24376, 12501, 17016, 11225, 17037, 24, 10983, 19254, 3927, 
	    23679, 13443, 16764, 24983, 10292, 3683, 9040, 8347, 3316, 8671, 378, 833, 5649, 
	    8573, 18503, 16032, 3320, 23758, 11986, 14276, 19482, 2100, 3152, 1581, 12783, 
	    22281, 2343, 13417, 4612, 3733, 14460, 24346, 17331, 7667, 9128, 15128, 11424, 
	    11800, 1135, 4913, 10868, 12726, 20558, 4403, 22054, 13053, 9371, 17542, 16199, 
	    24626, 16478, 1036, 8914, 5386, 14016, 12093, 7248, 19344, 16033, 7737, 11630, 
	    13040, 5358, 5950, 8117, 3008, 12296, 15080, 19070, 17702, 10510, 16911, 33, 
	    13159, 12846, 6791, 11942, 2709, 11526, 3425, 20540, 5629, 19801, 23943, 8990, 
	    6350, 8186, 19474, 23504, 18129, 16372, 450, 8435, 16679, 1998, 15351, 20755, 
	    14584, 1756, 14240, 17539, 13426, 24938, 4497, 16418, 12668, 13149, 14921, 4476, 
	    13223, 23272, 3173, 10912, 12467, 14230, 4486, 21464, 7186, 3722, 13786, 24343, 
	    7289, 4528, 12763, 19826, 7794, 7152, 10597, 1045, 2254, 4422, 4187, 1209, 
	    19716, 3831, 24764, 6091, 10048, 21679, 19850, 10008, 17641, 6594, 1521, 19150, 
	    11890, 22659, 5786, 9306, 1635, 21218, 14231, 2914, 9582, 12442, 3865, 10393, 
	    9547, 1259, 10460, 19430, 4395, 17259, 17117, 15398, 19871, 8563, 5498, 23225, 
	    11919, 537, 16977, 12271, 6727, 9320, 12120, 17605, 17717, 3509, 10406, 2322, 
	    2581, 21164, 8259, 14149, 8957, 23123, 1773, 11041, 24404, 21892, 4165, 21949, 
	    20317, 24468, 8949, 11432, 275, 15626, 22961, 18200, 19275, 24252, 441, 20897, 
	    842, 4386, 22334, 2399, 11667, 18850, 4141, 10573, 24908, 2831, 14145, 14342, 
	    18963, 9842, 7705, 5264, 20199, 4631, 16882, 15774, 3778, 1078, 8074, 12400, 
	    14976, 16708, 1434, 7921, 23521, 22803, 23754, 13453, 18783, 13735, 19000, 
	    11934, 18610, 8784, 23044, 13391, 7649, 16273, 20351, 21420, 15551, 6948, 12479, 
	    1413, 24062, 18593, 16207, 9840, 14045, 6619, 16056, 15429, 21625, 7168, 21651, 
	    4741, 22635, 4745, 6600, 21356, 24707, 22722, 14044, 3938, 14973, 12888, 16711, 
	    18595, 4997, 23094, 19837, 10047, 7167, 5580, 13996, 14375, 13330, 5736, 3614, 
	    9815, 10280, 8730, 16165, 7422, 1807, 17933, 8510, 12430, 12096, 18105, 9510, 
	    23853, 8866, 7445, 23203, 6257, 15750, 2295, 23717, 13293, 22606, 20226, 3782, 
	    21486, 3196, 21660, 16715, 16060, 3345, 23735, 16348, 19557, 21657, 13552, 632, 
	    19431, 4427, 19156, 10910, 21755, 4895, 19239, 15493, 10845, 9576, 18867, 19263, 
	    535, 20696, 14406, 23644, 10055, 10840, 13415, 13503, 5437, 17568, 11016, 3759, 
	    3436, 8660, 6056, 2365, 18159, 2559, 14242, 12854, 4746, 2843, 15599, 6045, 
	    24250, 11161, 17939, 24825, 7620, 4344, 24179, 21865, 13899, 16109, 16600, 
	    13610, 8429, 12209, 20619, 23515, 7219, 14687, 24809, 11997, 17254, 6167, 14665, 
	    22612, 2252, 4790, 9063, 23489, 10722, 2436, 6401, 20571, 24329, 11203, 15718, 
	    20659, 10719, 7276, 7464, 13022, 2708, 22535, 2818, 7710, 22294, 11968, 4202, 
	    20486, 10486, 1264, 15916, 12159, 2299, 11612, 2286, 9213, 8904, 12410, 20394, 
	    8091, 2151, 16439, 21209, 22636, 13505, 24655, 3398, 23359, 4416, 3230, 8958, 
	    417, 12565, 675, 12836, 2238, 21917, 8094, 17286, 6264, 11579, 17378, 4360, 
	    11300, 2511, 18731, 24703, 8222, 14458, 18922, 11656, 1367, 8378, 13009, 18944, 
	    14433, 13340, 5334, 1292, 13497, 17055, 21987, 18020, 15286, 18244, 17206, 
	    15790, 14486, 9211, 23622, 15865, 4748, 24140, 19813, 21605, 12162, 22291, 
	    24860, 908, 21225, 16363, 1866, 8960, 9349, 5059, 3961, 4099, 9731, 2625, 3648, 
	    4083, 13496, 17680, 22575, 14696, 13109, 6421, 12688, 10364, 15705, 18334, 5055, 
	    4996, 20954, 4661, 11186, 14162, 24344, 55, 17216, 18074, 18646, 13408, 19570, 
	    22442, 12328, 23759, 22634, 2132, 12217, 16952, 3348, 24084, 14861, 4831, 4833, 
	    8846, 11544, 24298, 16304, 22757, 11679, 14431, 4968, 22127, 6577, 15158, 10758, 
	    19538, 5958, 22608, 24294, 23077, 8699, 17121, 13488, 24355, 3944, 12931, 17828, 
	    8966, 1110, 21572, 524, 22705, 20691, 13226, 21430, 4089, 1503, 14659, 3629, 
	    17383, 20648, 18837, 11514, 9823, 2502, 10678, 12995, 14307, 21951, 1796, 20496, 
	    5151, 18589, 22031, 4722, 16503, 8064, 10389, 16760, 16524, 20345, 24046, 16464, 
	    13019, 1865, 11375, 9214, 5224, 19249, 17441, 12158, 15696, 20465, 17941, 11101, 
	    20670, 22497, 11334, 19535, 24959, 15691, 20152, 15118, 6510, 18952, 12063, 
	    24646, 1001, 24678, 2120, 20957, 12015, 22464, 10061, 4935, 16498, 16868, 2342, 
	    16696, 6247, 16114, 1861, 21009, 21429, 23445, 21487, 4593, 8325, 14239, 17160, 
	    8177, 12023, 2371, 17327, 19384, 5547, 8714, 24624, 14995, 17584, 23528, 14771, 
	    11277, 4749, 15623, 7420, 10555, 2359, 15963, 4924, 12247, 14950, 19835, 18560, 
	    23388, 18779, 19788, 14775, 23699, 11768, 8765, 22567, 23502, 11798, 12667, 
	    4242, 3202, 5031, 10343, 1033, 19710, 7729, 399, 19836, 23230, 12963, 2657, 
	    8816, 18066, 11284, 5058, 7011, 22901, 7393, 19972, 7606, 4409, 1314, 44, 22499, 
	    186, 802, 16879, 5466, 5683, 5656, 9516, 126, 1474, 22304, 16062, 17463, 5980, 
	    296, 7060, 22045, 17579, 9704, 1822, 18552, 24645, 8184, 10133, 8972, 24414, 
	    8341, 3573, 6989, 22770, 21678, 16640, 5846, 24367, 11020, 13873, 15161, 22532, 
	    9328, 16509, 20892, 5080, 15393, 11032, 22466, 16003, 17681, 13559, 4125, 10794, 
	    4088, 24208, 1008, 24577, 501, 5776, 24633, 11009, 95, 243, 22595, 15686, 14461, 
	    1607, 10447, 13756, 19211, 22259, 5791, 13339, 24804, 13660, 16030, 8964, 2509, 
	    7651, 5369, 3688, 24288, 10224, 14767, 16414, 15001, 2072, 2826, 24101, 9060, 
	    6563, 1213, 8311, 6591, 7236, 7058, 14450, 6615, 13738, 16172, 21301, 8152, 
	    17672, 3847, 24322, 22831, 7904, 17867, 2233, 3030, 21595, 23631, 21264, 21953, 
	    11567, 4311, 23841, 1788, 15786, 17021, 5659, 19589, 14122, 10204, 7728, 57, 
	    13681, 22293, 5694, 9578, 13927, 21196, 12607, 6994, 15757, 24552, 19126, 13244, 
	    18517, 13795, 20236, 23434, 11316, 11244, 6526, 19147, 10990, 9089, 3711, 8727, 
	    15034, 12293, 24162, 3915, 21398, 13726, 174, 17187, 13924, 21973, 6847, 21775, 
	    4218, 14220, 16376, 10769, 22005, 19391, 1416, 9651, 23605, 21270, 23281, 152, 
	    8201, 18903, 20202, 9282, 18162, 13175, 6144, 12988, 6375, 11486, 14383, 1599, 
	    9407, 13807, 12602, 9372, 14778, 21367, 12564, 8597, 72, 15684, 21503, 5227, 
	    20851, 6587, 11639, 2382, 22450, 21104, 1390, 24073, 8267, 2489, 7585, 6064, 
	    3283, 9773, 24151, 3299, 19971, 852, 15224, 1155, 22943, 22900, 15570, 20586, 
	    24586, 1842, 23297, 24109, 18748, 5311, 11072, 17820, 2506, 6516, 9828, 11447, 
	    18626, 14731, 2832, 729, 22646, 4540, 12992, 13793, 22719, 19143, 15602, 23126, 
	    24256, 7431, 20262, 20871, 10950, 15196, 19426, 14893, 23462, 23903, 2627, 
	    18728, 7753, 1907, 10554, 2113, 15148, 18012, 3525, 1699, 10566, 12799, 14430, 
	    9589, 18878, 24743, 16235, 4788, 17949, 1447, 14052, 7650, 24196, 751, 17233, 
	    13257, 9498, 3982, 18859, 22360, 6759, 2228, 19958, 8779, 15340, 16335, 24400, 
	    23071, 2695, 21880, 14004, 15662, 16838, 3811, 10349, 20693, 9029, 3776, 8051, 
	    6007, 13461, 17569, 4946, 1657, 7013, 16657, 575, 10536, 10523, 2413, 23697, 
	    1759, 16752, 12078, 20682, 13237, 12819, 17346, 23369, 8771, 9857, 20952, 12155, 
	    13556, 0, 3350, 13145, 24462, 17655, 15562, 15828, 9175, 2136, 7311, 12021, 
	    4760, 14485, 4348, 4875, 5814, 14813, 16204, 3337, 24194, 7825, 9558, 11045, 
	    16903, 6564, 13086, 23728, 3605, 16693, 12425, 7790, 8759, 5862, 23325, 12827, 
	    12582, 6445, 1648, 19901, 8236, 2022, 1329, 17919, 9771, 19480, 24314, 13724, 
	    15674, 6692, 3941, 2265, 17502, 14338, 15776, 11478, 22243, 19153, 2691, 22172, 
	    12863, 12295, 8825, 5267, 11786, 10302, 13460, 2915, 14938, 7875, 6732, 9359, 
	    18064, 19565, 9813, 13060, 13667, 5009, 905, 19047, 10866, 1933, 13835, 10011, 
	    18617, 2562, 8394, 16364, 9844, 8108, 21570, 13113, 16889, 16070, 24575, 22574, 
	    15272, 2617, 8806, 24290, 18492, 3261, 13301, 1976, 5602, 13295, 18153, 24965, 
	    22457, 2460, 24687, 8731, 3069, 17140, 23396, 5057, 1347, 8790, 20003, 11605, 
	    3341, 6870, 22872, 9786, 14848, 80, 17773, 4110, 8324, 12622, 16680, 3535, 4685, 
	    17468, 19830, 13003, 20490, 21888, 18157, 5010, 13337, 8722, 2005, 14453, 24309, 
	    24285, 8359, 8542, 21294, 20148, 14939, 3623, 216, 8591, 17424, 24226, 18502, 
	    8027, 16658, 1897, 367, 3896, 11575, 20971, 4391, 22034, 16701, 20557, 21317, 
	    1151, 13050, 4931, 2175, 8576, 24478, 5811, 21073, 10181, 12511, 23694, 15255, 
	    16850, 9969, 22325, 15360, 18562, 1217, 22904, 5717, 2980, 24628, 17761, 16795, 
	    3731, 13350, 7382, 18664, 19041, 6886, 16458, 18247, 1499, 16013, 6379, 8506, 
	    17377, 8768, 21692, 4674, 9825, 331, 1500, 16209, 5281, 24870, 19685, 15079, 
	    11124, 17455, 20395, 19030, 10841, 20946, 6533, 2180, 16630, 4818, 4275, 12341, 
	    23174, 2430, 4133, 9323, 18713, 14177, 19342, 20863, 24074, 18948, 8809, 11234, 
	    19439, 2147, 24569, 8254, 14009, 10800, 18276, 9431, 19922, 24826, 17875, 17644, 
	    9557, 18362, 6777, 14752, 24163, 9117, 14583, 1837, 3882, 5158, 20986, 7130, 
	    8144, 7208, 8032, 3494, 9008, 14698, 5449, 22638, 21690, 9598, 18520, 21536, 
	    3385, 22237, 23324, 22313, 17505, 19479, 9986, 8830, 24192, 15665, 16316, 13709, 
	    20090, 13941, 973, 2746, 6543, 6157, 22640, 9405, 9653, 2030, 14806, 16193, 
	    9024, 9908, 14951, 14814, 14690, 9954, 17193, 23428, 20818, 15988, 17298, 8578, 
	    9943, 21414, 3354, 18907, 13653, 7128, 20532, 8593, 14356, 2315, 17400, 9551, 
	    2344, 3312, 24594, 23606, 17706, 2937, 23817, 9152, 11729, 11366, 8755, 20349, 
	    7143, 1968, 7312, 7081, 20951, 17657, 18124, 13777, 21345, 16311, 21247, 21460, 
	    11893, 20108, 24709, 11721, 24953, 15555, 17170, 24228, 10672, 10593, 19055, 
	    8420, 24664, 10110, 9534, 5418, 7553, 13158, 3148, 3801, 7493, 5645, 18370, 
	    6018, 23670, 8538, 12330, 5435, 5004, 7411, 14241, 13548, 5703, 5119, 15315, 
	    4465, 11281, 120, 11265, 14632, 17784, 6183, 23570, 16864, 22215, 10461, 19893, 
	    13143, 183, 13613, 21030, 1568, 15872, 11047, 23138, 9976, 22821, 919, 5443, 
	    16653, 5961, 4699, 22462, 23673, 14804, 24452, 20319, 24186, 5677, 17821, 21007, 
	    24846, 8274, 23611, 20658, 12119, 18933, 19064, 1091, 12557, 9384, 24471, 13384, 
	    19506, 19926, 8498, 12837, 5278, 8010, 16025, 18624, 23389, 18495, 6810, 17948, 
	    13767, 14713, 7596, 14980, 9667, 212, 19243, 21639, 23905, 22571, 16237, 4154, 
	    10015, 23365, 20409, 17786, 23168, 23073, 15713, 9841, 17358, 9802, 11996, 
	    20797, 14065, 16773, 9751, 22625, 19990, 18183, 9392, 10372, 10863, 16413, 
	    19252, 5569, 5481, 23116, 8393, 1955, 2899, 4888, 2310, 5825, 18823, 7307, 7671, 
	    18346, 2944, 20058, 6752, 12310, 9537, 13071, 13338, 2149, 19417, 17974, 1241, 
	    14864, 18759, 15374, 23385, 7475, 896, 21985, 4393, 16616, 8523, 20529, 22225, 
	    1007, 22193, 20249, 23671, 4743, 18929, 23150, 4330, 13094, 11459, 21842, 12225, 
	    20350, 21240, 10210, 8140, 18980, 15179, 7157, 14844, 3198, 3794, 12964, 8976, 
	    16802, 15576, 1465, 14615, 20808, 1204, 3838, 17419, 15671, 16067, 4151, 24745, 
	    9426, 20473, 16082, 13800, 12958, 22289, 18485, 17808, 16504, 9021, 24529, 7346, 
	    7259, 22493, 16108, 2197, 2541, 22189, 6967, 15282, 17701, 20164, 9203, 720, 
	    10838, 3333, 5476, 21353, 1934, 16871, 24814, 6991, 435, 1387, 11253, 24156, 
	    4012, 4537, 8305, 9471, 1074, 256, 1319, 1790, 22688, 20269, 12360, 8522, 10229, 
	    11017, 3805, 15835, 2293, 16724, 15592, 11713, 24931, 15441, 24506, 23945, 8833, 
	    14456, 4609, 4127, 287, 23330, 13976, 13824, 22555, 8136, 23028, 231, 6259, 
	    2340, 23808, 18802, 1548, 19010, 19009, 24756, 18668, 12969, 23836, 81, 21239, 
	    9171, 18322, 23864, 21468, 23001, 14325, 6860, 14243, 11082, 12432, 23261, 9333, 
	    21803, 17466, 20039, 618, 2496, 6397, 24863, 11640, 21508, 8642, 12683, 9224, 
	    19169, 8304, 20937, 14599, 12121, 16342, 7398, 1588, 21050, 23373, 20088, 16020, 
	    9421, 17308, 7774, 17812, 17492, 11427, 22783, 20354, 17926, 3548, 22658, 24052, 
	    5486, 20717, 5221, 9486, 21103, 15210, 7195, 13968, 3803, 8546, 1709, 24276, 
	    20977, 8754, 20308, 20112, 5187, 23283, 19682, 7600, 10125, 17566, 7747, 23280, 
	    12423, 13250, 11923, 13637, 2848, 12019, 11627, 7282, 11695, 6327, 10499, 10561, 
	    4667, 13289, 17061, 3985, 22995, 11218, 13829, 22657, 21167, 3334, 4245, 2769, 
	    12906, 1816, 12202, 12554, 8490, 17741, 23551, 13518, 16465, 1557, 14987, 8392, 
	    13473, 12611, 13274, 7693, 16644, 3478, 21252, 10563, 10476, 10407, 20298, 
	    10626, 10735, 8053, 16322, 5121, 4175, 20616, 22594, 18996, 10384, 24379, 8938, 
	    11188, 3432, 9487, 23547, 4689, 11935, 14824, 5854, 23176, 20758, 13953, 22845, 
	    8080, 23838, 3612, 8633, 10896, 3758, 23751, 13666, 7923, 916, 11794, 20573, 
	    10705, 7404, 15207, 2851, 2090, 2276, 8209, 23803, 12873, 22484, 4203, 12982, 
	    3260, 17000, 10088, 6283, 22087, 18193, 20320, 4138, 8277, 5184, 12805, 12517, 
	    14774, 14279, 6997, 23175, 6378, 7736, 21700, 22221, 5759, 8629, 10283, 19463, 
	    3423, 6388, 20539, 21423, 11770, 6785, 13347, 20219, 2779, 21179, 14291, 23958, 
	    12938, 2842, 6706, 21314, 18261, 7263, 4850, 9997, 24270, 9065, 23753, 22515, 
	    18949, 16131, 11282, 15673, 6834, 796, 23012, 22140, 5249, 9092, 23018, 8472, 
	    24557, 10359, 19930, 9604, 22846, 13509, 13720, 20821, 24937, 15731, 14035, 
	    17134, 10273, 8651, 8242, 14591, 801, 22504, 10507, 20792, 19380, 13011, 9559, 
	    22736, 22012, 13676, 1594, 14043, 12315, 13935, 15478, 5675, 18161, 11351, 1813, 
	    1077, 6879, 21071, 14068, 21108, 15157, 16350, 17242, 7619, 11324, 480, 12999, 
	    5434, 8706, 4109, 3279, 14051, 22132, 12156, 13367, 13218, 4041, 11520, 2484, 
	    3036, 18117, 22417, 10888, 12345, 19008, 20678, 23636, 5250, 3369, 14961, 21656, 
	    17800, 20472, 10092, 15514, 7878, 4967, 20643, 22662, 17617, 21274, 9816, 17618, 
	    4338, 19941, 11367, 24112, 17199, 11473, 834, 16678, 8462, 7180, 20054, 13542, 
	    4720, 5616, 14614, 19189, 18126, 7571, 23454, 1775, 13035, 22396, 16549, 24430, 
	    12243, 1438, 13378, 15544, 18746, 6358, 20463, 16996, 5460, 23483, 11533, 18990, 
	    16863, 24446, 1118, 23677, 18688, 10261, 18590, 7165, 17755, 22556, 11046, 
	    14866, 5891, 5245, 18357, 23210, 1227, 18043, 11995, 13303, 681, 12897, 3553, 
	    23506, 6466, 374, 16024, 23215, 15870, 20279, 8391, 911, 20955, 9111, 7121, 
	    9799, 1018, 20067, 10592, 18858, 11388, 18262, 765, 5565, 4721, 17587, 24258, 
	    14404, 19377, 19279, 14053, 11294, 23729, 9273, 17376, 7122, 23033, 10991, 
	    17116, 17537, 19695, 18521, 2179, 4877, 13455, 5528, 11738, 23564, 9570, 15976, 
	    23468, 24427, 8021, 24508, 4413, 4846, 14877, 7291, 22356, 4463, 9194, 20560, 
	    15930, 6215, 6115, 21735, 19887, 20225, 13529, 23745, 24432, 14390, 4417, 5408, 
	    13104, 10824, 7160, 10716, 5459, 13853, 14285, 22998, 19623, 15965, 298, 586, 
	    12365, 10612, 13070, 23518, 7523, 15283, 17986, 3616, 24927, 5092, 15439, 8917, 
	    2786, 15787, 8276, 13041, 7279, 16720, 23179, 16961, 18191, 23275, 15960, 10680, 
	    16268, 20305, 3989, 15445, 19627, 23057, 4129, 10347, 23342, 16284, 7139, 13717, 
	    16922, 24776, 9966, 1238, 21496, 1087, 24596, 1930, 24485, 2962, 20636, 6303, 
	    12871, 18550, 12060, 22505, 20734, 1454, 23463, 1289, 24915, 6146, 12998, 4645, 
	    1937, 624, 23312, 17613, 22938, 18760, 15215, 9341, 3960, 1328, 22703, 10935, 
	    8168, 10284, 12435, 17728, 14039, 4866, 19532, 813, 21707, 12939, 24785, 16370, 
	    2247, 22088, 19414, 14888, 677, 21337, 1466, 11496, 12117, 11834, 23791, 15335, 
	    3496, 5682, 3331, 1794, 9792, 21263, 6057, 7030, 15183, 20817, 17593, 14609, 
	    24111, 8231, 19809, 8669, 12126, 164, 13521, 14934, 14592, 10019, 11058, 17191, 
	    10078, 17020, 5022, 20589, 6428, 3189, 17900, 6781, 7159, 19935, 21998, 2703, 
	    464, 23213, 14664, 7538, 16183, 23305, 8983, 5783, 11916, 4651, 7218, 3286, 
	    3268, 3491, 19839, 20720, 4578, 19878, 2921, 12635, 24944, 9619, 1282, 5205, 
	    6848, 24899, 15871, 3823, 14677, 10654, 19680, 13625, 21852, 24461, 283, 1550, 
	    21676, 3277, 23678, 8088, 16571, 10226, 14312, 16377, 18671, 17384, 2727, 15300, 
	    12372, 18723, 792, 22866, 19007, 11518, 14, 4223, 18230, 19860, 902, 17667, 
	    21311, 19563, 9204, 5933, 14952, 22889, 14248, 15111, 7135, 11474, 14903, 20220, 
	    14669, 221, 773, 14621, 13234, 11220, 8301, 17427, 9443, 3317, 11123, 19843, 
	    2845, 17705, 3591, 22713, 3239, 2188, 2013, 13447, 5951, 564, 13423, 11088, 
	    20020, 9552, 18379, 17775, 18791, 23672, 9654, 22874, 4327, 1660, 7226, 22452, 
	    8702, 6744, 10791, 4257, 20837, 24284, 21896, 13117, 24882, 1694, 3081, 20412, 
	    15837, 9262, 8877, 3980, 17658, 10833, 14070, 11067, 5554, 9007, 16466, 6645, 
	    15977, 23182, 17734, 230, 23793, 13537, 19590, 2684, 12255, 17087, 516, 6961, 
	    19831, 22911, 9649, 14516, 10266, 1579, 4950, 22950, 10444, 19719, 21977, 12196, 
	    17437, 8437, 6150, 712, 2589, 19937, 16473, 16069, 16890, 22305, 20316, 19640, 
	    13438, 8119, 11553, 12573, 8417, 9887, 21727, 9795, 24960, 24793, 1510, 2355, 
	    3226, 6654, 3715, 16041, 14479, 232, 7380, 21637, 23941, 8134, 12760, 16617, 
	    10342, 11623, 3839, 6079, 2046, 8308, 21610, 14138, 24949, 23780, 22136, 15794, 
	    19585, 7367, 6918, 16038, 10443, 15498, 13486, 16437, 20083, 15373, 741, 19869, 
	    7827, 18336, 16291, 1159, 12775, 10599, 644, 23919, 19020, 11952, 9164, 6276, 
	    4695, 5333, 2168, 16835, 23074, 2237, 2733, 4239, 20888, 23169, 13480, 17103, 
	    2034, 19376, 14568, 5071, 7928, 10969, 4585, 9790, 22311, 5020, 12701, 19360, 
	    24352, 10274, 3737, 16574, 20960, 18732, 8962, 487, 4921, 21492, 2582, 19555, 
	    2865, 18323, 18689, 3736, 22267, 16645, 7325, 15625, 2071, 22066, 22208, 16027, 
	    11516, 3757, 22079, 20868, 9607, 14830, 22011, 18381, 10132, 4139, 21906, 2595, 
	    13155, 16206, 5646, 11900, 5636, 19320, 3014, 13674, 15352, 391, 23565, 17368, 
	    7342, 23824, 18241, 2668, 23830, 22817, 12719, 21329, 20681, 5637, 8696, 14997, 
	    2629, 21359, 4659, 5915, 21361, 13615, 11887, 13284, 20774, 20143, 11420, 2788, 
	    3684, 22180, 22195, 21010, 9330, 21718, 13659, 9783, 10691, 19807, 10454, 14530, 
	    3482, 13963, 405, 9695, 11803, 9485, 16620, 14985, 14859, 18062, 12253, 1580, 
	    7666, 10081, 10087, 14913, 4706, 22185, 5104, 23439, 8055, 16221, 8338, 22823, 
	    12408, 8082, 23347, 12648, 21887, 22768, 11355, 11714, 23767, 14596, 19676, 
	    10130, 18686, 4076, 15663, 24815, 7414, 5879, 5043, 12446, 11006, 24493, 11012, 
	    936, 16559, 21537, 13082, 18807, 4636, 22809, 21907, 14629, 14715, 14821, 17007, 
	    1472, 9028, 10773, 1723, 13591, 63, 15365, 22231, 10286, 16309, 416, 13134, 
	    7903, 22737, 17205, 9299, 16676, 978, 23480, 8843, 9588, 2782, 14476, 10370, 
	    1285, 17627, 11256, 21409, 19605, 19415, 4074, 21710, 9599, 21178, 18265, 19322, 
	    12908, 1726, 2006, 6447, 22309, 7520, 526, 22437, 13911, 22395, 6070, 5223, 
	    7615, 15506, 1880, 2612, 5286, 2088, 12590, 8315, 3827, 2557, 18086, 19656, 
	    24334, 7508, 9265, 24982, 9225, 7593, 24723, 23124, 22258, 22971, 7837, 2199, 
	    261, 20991, 6314, 14930, 19503, 1873, 13772, 13828, 14542, 5139, 14758, 1411, 
	    16771, 1782, 8863, 11698, 17387, 18249, 8271, 24602, 3681, 20430, 21233, 21562, 
	    12559, 17363, 22833, 17659, 7818, 7371, 15237, 15844, 21469, 693, 18061, 22587, 
	    20092, 5204, 24187, 10811, 303, 21989, 20609, 415, 20280, 645, 7487, 4073, 
	    21992, 4995, 23990, 12505, 21457, 6110, 4589, 1645, 8873, 7138, 7271, 15334, 
	    2999, 10033, 22019, 15114, 19592, 20692, 8287, 12106, 2642, 18567, 12490, 17484, 
	    24970, 579, 14165, 19633, 7402, 12168, 1835, 20434, 20303, 300, 4064, 18327, 
	    24259, 153, 22970, 7045, 20620, 18915, 15856, 24795, 643, 4231, 3017, 8775, 
	    23392, 22234, 7372, 5618, 19670, 2137, 21145, 14094, 14749, 4980, 9631, 7246, 
	    10265, 9074, 11400, 2644, 12673, 6964, 13365, 12883, 15664, 13809, 19244, 14121, 
	    24333, 18765, 20091, 9414, 19780, 24855, 15896, 24913, 4989, 22927, 2289, 2877, 
	    15841, 10528, 17215, 13619, 4958, 7977, 5198, 2023, 13459, 15075, 7151, 5319, 
	    18427, 22786, 5795, 14766, 11914, 17762, 14060, 23189, 16189, 21410, 1440, 
	    22254, 18144, 19679, 17562, 18586, 14805, 23789, 20506, 10958, 11254, 21787, 
	    22761, 697, 16349, 20887, 22513, 2755, 14650, 10077, 7032, 12421, 2110, 2163, 
	    20520, 24700, 1073, 22772, 9584, 17325, 12412, 17375, 7975, 16956, 4438, 9899, 
	    18932, 14169, 11094, 9335, 22272, 17198, 11918, 850, 1035, 9377, 5033, 5667, 
	    8608, 9173, 20074, 11696, 9971, 5176, 7550, 22040, 5118, 310, 13475, 21160, 
	    22827, 3998, 6013, 24875, 23490, 16786, 12839, 1196, 2087, 12503, 24424, 1650, 
	    10064, 4183, 1239, 21477, 13061, 16594, 10141, 19375, 16004, 14185, 20622, 8994, 
	    701, 4711, 22033, 3280, 24239, 3952, 22461, 21408, 24612, 17015, 11167, 13564, 
	    8552, 466, 1505, 4331, 24369, 5063, 10206, 19136, 4408, 7185, 16129, 9784, 
	    17922, 9890, 1854, 13510, 2714, 10865, 2519, 4059, 5613, 12494, 4514, 23499, 
	    4280, 18829, 11519, 12266, 15418, 3335, 6809, 11851, 1324, 18227, 3764, 17362, 
	    270, 5708, 18788, 23683, 24545, 24548, 19448, 9702, 15922, 6950, 10325, 10957, 
	    17086, 10236, 22129, 3249, 2348, 13343, 213, 23975, 11063, 19951, 10138, 19720, 
	    11654, 19409, 7742, 1733, 20238, 13730, 13513, 23006, 20424, 10829, 23823, 6988, 
	    19699, 8792, 21275, 13700, 12556, 6901, 11064, 17323, 12419, 14255, 1793, 12773, 
	    17072, 7212, 20069, 306, 3181, 22433, 5533, 24753, 1655, 24675, 2462, 14502, 
	    6003, 21952, 20915, 5784, 626, 1394, 6016, 8757, 14294, 8239, 7873, 10026, 3112, 
	    20475, 12001, 18055, 7570, 16021, 10199, 13500, 12737, 10670, 2515, 23747, 
	    18219, 18483, 9740, 12519, 16264, 19232, 24932, 20193, 6015, 2183, 18177, 21131, 
	    24266, 19390, 24672, 8634, 15031, 14931, 22678, 19368, 19194, 17176, 3002, 
	    17128, 3894, 6422, 24148, 8384, 18745, 1893, 17054, 9013, 13903, 3977, 622, 
	    20753, 19553, 14842, 2078, 3116, 1079, 10906, 22064, 8743, 9953, 21431, 5425, 
	    13590, 12346, 1185, 20482, 20140, 3292, 18120, 22890, 24535, 13991, 19881, 1029, 
	    3871, 23885, 7711, 14249, 13568, 10883, 11107, 6976, 14517, 11248, 19540, 16964, 
	    18155, 23381, 822, 6325, 17096, 6907, 11189, 18968, 8419, 9129, 24419, 3197, 
	    9209, 16286, 5742, 9073, 8807, 4709, 9523, 11031, 13583, 11671, 17196, 20228, 
	    7383, 2060, 11670, 4033, 18722, 15349, 20072, 23181, 10760, 11392, 7119, 11430, 
	    17558, 4002, 2958, 10067, 18134, 15330, 13535, 5500, 19108, 11163, 19968, 1950, 
	    18598, 7961, 8484, 13208, 16649, 13465, 7363, 5307, 5816, 206, 10263, 16095, 
	    12027, 13144, 24100, 22046, 13064, 11866, 12174, 20043, 20288, 1704, 4619, 
	    16510, 21930, 21621, 13885, 2634, 234, 2752, 1624, 8810, 14452, 24229, 14764, 
	    12337, 3311, 13517, 22855, 10399, 8707, 21338, 528, 5815, 23698, 13748, 8682, 
	    17118, 18538, 18067, 17520, 20028, 14511, 14047, 9633, 2504, 2508, 156, 22469, 
	    7155, 4051, 5471, 19085, 12865, 15812, 23433, 3314, 9365, 17219, 9933, 23866, 
	    6926, 18459, 2594, 22401, 172, 9354, 21445, 12899, 8456, 7007, 16486, 18416, 
	    12402, 14022, 2991, 652, 11507, 20867, 14564, 24469, 12798, 2275, 7945, 2763, 
	    19114, 10583, 8019, 7499, 17725, 23337, 14063, 16395, 15629, 11069, 9283, 20950, 
	    15830, 10217, 22836, 20921, 18145, 15536, 18603, 18349, 11004, 20786, 10002, 
	    5426, 6754, 22954, 5054, 20724, 10679, 19859, 10155, 6509, 2107, 13334, 1522, 
	    7795, 15657, 7764, 10244, 21687, 4235, 3743, 15909, 914, 133, 24212, 13768, 
	    3359, 20179, 12188, 1679, 11209, 9850, 19924, 14003, 2448, 8215, 24853, 14499, 
	    5928, 19810, 5506, 5258, 15784, 7435, 6818, 13135, 18192, 5168, 18326, 22712, 
	    1318, 20832, 11689, 6026, 21365, 17888, 10243, 12128, 8465, 687, 18210, 5627, 
	    17651, 385, 19880, 16332, 22914, 1852, 23234, 16803, 8360, 24299, 3932, 18785, 
	    14941, 24896, 325, 18632, 23314, 22230, 12343, 11561, 10518, 14286, 20427, 
	    12978, 19394, 22519, 11979, 328, 21646, 23146, 18147, 7385, 22700, 21650, 23088, 
	    13553, 15378, 22102, 9774, 5598, 21306, 3679, 10201, 13769, 9316, 15642, 694, 
	    937, 4410, 16638, 21235, 6726, 9477, 16219, 20374, 4878, 12833, 4873, 12625, 
	    7004, 2630, 9614, 18118, 23343, 20638, 9800, 7379, 10574, 17042, 12641, 21877, 
	    13743, 15427, 4441, 15120, 22397, 24122, 4569, 22650, 15406, 369, 16457, 23566, 
	    13877, 12624, 8864, 1561, 24856, 12756, 10986, 9419, 3133, 5888, 3029, 13116, 
	    7689, 2870, 6833, 3444, 12618, 1175, 14235, 11868, 10218, 18033, 595, 23654, 
	    7273, 10085, 724, 11219, 11559, 6559, 19572, 13103, 7403, 16902, 23750, 8290, 
	    3474, 3903, 6678, 6255, 24742, 295, 9272, 7738, 16704, 8887, 17223, 17465, 1678, 
	    22156, 17876, 23288, 12679, 11663, 6153, 22444, 16659, 22764, 12489, 15100, 
	    20272, 1547, 10522, 13782, 17396, 21833, 8635, 5954, 4900, 2084, 1797, 16118, 
	    23258, 21728, 17294, 14031, 5625, 4627, 3153, 805, 5365, 4251, 18361, 19730, 
	    569, 21116, 21088, 5911, 5014, 1478, 10788, 19696, 3772, 18140, 14033, 20999, 
	    14267, 14075, 11139, 11874, 6943, 8978, 17972, 22095, 12278, 17475, 12809, 
	    10353, 5, 2433, 24761, 14705, 10432, 8093, 21217, 9526, 11608, 13622, 113, 
	    13844, 8188, 14262, 11568, 12068, 11246, 6275, 12606, 8965, 8620, 19236, 19398, 
	    14021, 16615, 600, 3483, 8521, 15469, 24059, 20719, 16269, 16379, 7354, 1333, 
	    23160, 12157, 4763, 16333, 5909, 6919, 17819, 10267, 9097, 20789, 21045, 16883, 
	    6685, 255, 24365, 14194, 7746, 17834, 10816, 18424, 16159, 13034, 5101, 15688, 
	    23576, 14802, 9009, 3009, 20162, 3969, 15197, 18579, 10253, 23437, 19741, 21875, 
	    4825, 24267, 22871, 19852, 16860, 7096, 16136, 17483, 20856, 11243, 3502, 21668, 
	    24054, 13906, 23289, 12593, 9398, 18329, 7094, 14163, 2871, 21135, 11985, 11863, 
	    12036, 1274, 10231, 24720, 9665, 11111, 6735, 24980, 5407, 4614, 1363, 16906, 
	    5687, 6425, 14088, 24659, 2317, 15723, 5615, 18581, 12055, 17571, 22801, 9990, 
	    14071, 24630, 24691, 10570, 12644, 13349, 15305, 5293, 238, 2689, 7516, 8910, 
	    1975, 1260, 1541, 10831, 2383, 7042, 8037, 13658, 11703, 22438, 17959, 1212, 
	    16806, 766, 19253, 6668, 8619, 17080, 13901, 12207, 445, 18820, 9132, 7722, 
	    3607, 19842, 12947, 8408, 5335, 23066, 10214, 10562, 11910, 15187, 16220, 1490, 
	    3145, 5420, 21138, 13422, 476, 22229, 21797, 13628, 18794, 3695, 3347, 20124, 
	    9218, 19002, 7924, 18555, 3440, 2138, 19704, 19364, 7898, 6772, 4598, 5174, 
	    2561, 17074, 11681, 13612, 22834, 19879, 17420, 9260, 8330, 23715, 23370, 19700, 
	    14448, 4456, 22142, 18815, 11021, 9155, 1071, 13921, 821, 4204, 12499, 1121, 
	    5297, 6253, 12362, 7723, 1644, 18778, 10269, 6040, 20208, 12200, 3901, 12460, 
	    8675, 11547, 12921, 24819, 16595, 6640, 2789, 12148, 7614, 22607, 19608, 13254, 
	    4501, 15040, 13429, 18213, 10079, 24238, 23645, 9281, 11730, 18519, 15952, 3561, 
	    9646, 15454, 21472, 1696, 7589, 18167, 1745, 17393, 23664, 22523, 7407, 10230, 
	    15208, 11097, 6764, 6122, 14150, 10321, 19097, 9843, 16808, 22430, 14808, 6086, 
	    13477, 3868, 903, 21377, 7177, 2690, 7339, 21253, 17857, 18693, 14841, 16648, 
	    20023, 658, 19877, 9068, 7761, 9090, 9389, 16178, 24869, 15155, 21644, 20785, 
	    18588, 23152, 22778, 10755, 20421, 19176, 20410, 9562, 39, 9435, 22060, 15294, 
	    11718, 3388, 3454, 10997, 15644, 16016, 10456, 627, 4326, 13817, 18348, 4928, 
	    11822, 16009, 20902, 12485, 12729, 8815, 8497, 18881, 14645, 14062, 3832, 7661, 
	    15983, 13947, 913, 14424, 11959, 24154, 1483, 14343, 14141, 11804, 3135, 17504, 
	    20630, 5661, 5944, 24984, 12171, 15651, 279, 20104, 7560, 14142, 18655, 23416, 
	    14281, 22159, 12657, 8358, 12169, 12210, 19748, 1081, 23989, 22472, 5929, 15212, 
	    24666, 2903, 3610, 13886, 8421, 10351, 10594, 12189, 13273, 23021, 22741, 16866, 
	    8445, 13705, 7429, 24518, 20820, 9318, 4712, 10857, 20862, 17224, 980, 6184, 
	    14113, 22130, 7809, 9288, 15511, 8649, 23730, 23045, 5810, 4005, 10681, 14393, 
	    6508, 12662, 1959, 18282, 23308, 13240, 21333, 9072, 6778, 20040, 8180, 18102, 
	    5322, 17107, 17043, 14894, 24505, 6252, 12589, 3593, 13404, 4226, 18458, 17853, 
	    23326, 267, 3732, 3590, 10458, 18313, 1397, 5148, 5553, 5686, 10836, 8511, 5030, 
	    3015, 2061, 18078, 15016, 2846, 8240, 20293, 6889, 23819, 8543, 11165, 10377, 
	    19274, 13508, 2324, 13586, 9251, 2176, 15239, 12465, 11042, 14134, 3646, 3096, 
	    24934, 11524, 5650, 3315, 5672, 4332, 10239, 819, 14707, 13298, 3289, 10804, 
	    24264, 1039, 24361, 1365, 13484, 3997, 276, 2414, 8011, 22730, 7343, 9026, 
	    11000, 14222, 14857, 2604, 7188, 7534, 1719, 5147, 10142, 17202, 8137, 20452, 
	    23399, 23091, 20654, 7387, 6212, 17266, 4435, 865, 23498, 24638, 8426, 16238, 
	    16000, 15945, 14127, 13382, 23600, 10987, 7235, 19603, 5606, 19044, 5018, 24708, 
	    17203, 10007, 14852, 12305, 4897, 9968, 23250, 8740, 7879, 147, 534, 21922, 
	    12920, 7117, 18614, 18961, 1102, 8460, 18132, 5084, 22244, 14722, 23332, 18763, 
	    1208, 18830, 22534, 19900, 17626, 7169, 21067, 9610, 15746, 8382, 23618, 1886, 
	    5976, 18141, 6324, 20150, 17612, 5073, 21730, 10702, 1869, 17838, 2246, 20192, 
	    6787, 18900, 5921, 6882, 13018, 22648, 8278, 14701, 3437, 21524, 2701, 13837, 
	    14457, 18700, 20437, 5993, 24536, 10408, 14464, 24698, 8921, 13068, 19756, 
	    21849, 9179, 13481, 10094, 14709, 21387, 17693, 1540, 15395, 20760, 3309, 7209, 
	    19822, 10693, 23375, 23899, 19302, 4867, 1431, 11406, 4618, 13827, 21246, 2174, 
	    17906, 10537, 11503, 10894, 11970, 12478, 20239, 3276, 10429, 7008, 18569, 
	    22628, 17567, 5570, 21863, 24854, 4551, 7727, 22835, 6449, 6780, 13981, 17385, 
	    7500, 7033, 21122, 16830, 22524, 9453, 1518, 9500, 17052, 15611, 15903, 19151, 
	    7187, 2835, 4288, 4362, 2076, 5642, 49, 20234, 6987, 4620, 5368, 8270, 17139, 
	    17946, 3855, 11811, 3393, 111, 23652, 20364, 13986, 19130, 281, 3042, 1853, 
	    8878, 21308, 67, 7521, 23807, 23316, 20373, 16974, 9655, 15375, 16397, 4212, 
	    979, 16198, 18855, 17410, 2368, 16134, 15566, 2712, 21706, 2652, 5126, 16561, 
	    19527, 22383, 13088, 3625, 8381, 6883, 1402, 4287, 21363, 14348, 3364, 10115, 
	    10862, 19952, 5354, 13433, 414, 10545, 10539, 19734, 22013, 12110, 16937, 2995, 
	    8157, 13852, 2957, 597, 8528, 17917, 23478, 23965, 15590, 4944, 12483, 10853, 
	    18372, 17785, 15532, 2945, 16713, 16404, 18017, 2184, 13006, 1435, 3339, 24778, 
	    18680, 15042, 19799, 17180, 19582, 16912, 21279, 23344, 11764, 24961, 17983, 
	    12282, 7775, 6960, 7239, 17835, 9271, 10688, 11096, 6181, 24881, 7877, 10571, 
	    14492, 12632, 14940, 18908, 10665, 1659, 11684, 5968, 9492, 4964, 22676, 5111, 
	    20707, 7758, 24515, 16312, 3104, 7296, 8364, 15546, 5599, 5770, 19866, 14359, 
	    19558, 17071, 394, 500, 12826, 20433, 5818, 3055, 2285, 19915, 6038, 23143, 
	    7734, 13846, 4680, 13194, 21330, 9297, 22235, 4953, 2547, 10506, 13602, 4433, 
	    9149, 7717, 1519, 5000, 17394, 21336, 11973, 16829, 15465, 20483, 18516, 11211, 
	    19634, 6274, 10817, 3420, 19609, 16758, 4731, 3550, 23593, 18022, 21323, 8131, 
	    22687, 4591, 23217, 2971, 9914, 9446, 9474, 4134, 20865, 13025, 8211, 6618, 
	    16527, 11622, 11856, 5535, 6699, 3705, 20839, 2041, 15574, 24381, 17790, 2224, 
	    12307, 4241, 15227, 22027, 15185, 7352, 17669, 1918, 11389, 16495, 5719, 8897, 
	    14166, 20940, 6887, 23371, 6975, 6187, 16289, 20100, 1376, 9151, 9707, 22928, 
	    23063, 10271, 21284, 494, 20699, 16137, 18130, 2445, 5880, 13854, 24029, 2311, 
	    17947, 10105, 16733, 8164, 683, 8736, 20156, 19052, 20639, 18997, 23979, 10916, 
	    5352, 22277, 16548, 24447, 9768, 5853, 3418, 22597, 4299, 19018, 5987, 12945, 
	    3210, 2986, 23131, 9440, 11723, 4008, 6661, 12957, 5068, 2759, 5089, 22381, 
	    1761, 16083, 1769, 5112, 3810, 4440, 1010, 23394, 19179, 4009, 17843, 12810, 
	    20307, 10378, 7441, 9137, 6113, 18398, 18021, 14014, 1080, 184, 11837, 18447, 
	    101, 16058, 23481, 10964, 3434, 4545, 17258, 14429, 8109, 90, 19399, 20369, 
	    21211, 6326, 4819, 9837, 23440, 8845, 20206, 1530, 13352, 8285, 5005, 21671, 
	    20555, 2806, 1183, 7766, 11302, 17354, 4126, 5078, 2472, 23139, 14329, 2000, 
	    23329, 2749, 11112, 15251, 248, 12997, 13563, 19501, 5772, 3477, 18114, 2397, 
	    8077, 16048, 6536, 20440, 6931, 21462, 23517, 7015, 20035, 14365, 22146, 7321, 
	    23438, 9245, 7786, 22981, 20060, 4150, 15559, 19828, 6768, 12583, 21547, 1777, 
	    22747, 23630, 3653, 7697, 6343, 7129, 16584, 14308, 19255, 22920, 22804, 23075, 
	    12144, 8647, 8017, 6462, 23719, 23929, 15156, 16598, 19774, 12306, 3532, 14561, 
	    4686, 3817, 1226, 10874, 121, 3438, 7754, 18544, 13375, 14330, 14023, 4027, 785, 
	    14656, 1037, 14306, 22580, 38, 19443, 13746, 9206, 2230, 20791, 24728, 9112, 
	    9730, 6972, 5709, 2700, 12587, 22756, 21680, 6504, 18127, 18832, 150, 4265, 
	    6690, 1108, 3205, 23996, 15276, 2513, 5690, 19545, 14413, 11127, 5732, 10757, 
	    11586, 11777, 5397, 11443, 2897, 1205, 5856, 15704, 20513, 4462, 16005, 11624, 
	    2550, 15205, 3760, 16805, 1244, 22273, 83, 5715, 11522, 18148, 22426, 16847, 
	    3796, 581, 1615, 10989, 9287, 9868, 9509, 14994, 18919, 5593, 1754, 20057, 
	    15408, 24951, 8970, 12521, 1774, 12968, 18332, 24383, 18321, 18601, 1441, 11415, 
	    3708, 7663, 7254, 10621, 20907, 14792, 9801, 18073, 6298, 7988, 7874, 827, 
	    21447, 19938, 2292, 13734, 15402, 21874, 17083, 7848, 13356, 8678, 24358, 9136, 
	    355, 18487, 7696, 1687, 7741, 14876, 21968, 6497, 6053, 17521, 24591, 21920, 
	    8565, 7781, 2852, 11562, 451, 4780, 13377, 17518, 13057, 21980, 14702, 20968, 
	    1348, 16064, 15184, 560, 13685, 5105, 15056, 6917, 16391, 2049, 5244, 3005, 
	    13910, 19311, 18263, 3475, 2029, 10481, 23060, 22115, 3908, 13866, 8548, 2469, 
	    625, 12942, 24032, 1346, 13956, 21711, 15894, 5161, 13665, 11442, 12981, 7106, 
	    5302, 1419, 16017, 7345, 12457, 13229, 7548, 23111, 2531, 18905, 2837, 16609, 
	    10035, 4484, 17908, 9981, 9785, 9612, 12599, 2032, 24527, 16788, 2185, 7220, 
	    21983, 18861, 10967, 14982, 7146, 7782, 12206, 17737, 269, 12056, 6209, 40, 
	    7019, 8040, 21342, 173, 5327, 4632, 7720, 9555, 24818, 7838, 8273, 22642, 10177, 
	    1358, 7785, 8438, 16709, 10127, 22619, 16612, 7247, 14948, 601, 4107, 13737, 
	    17487, 6812, 10395, 396, 2855, 21112, 18432, 24642, 17069, 23512, 15739, 12830, 
	    2777, 4851, 21719, 7643, 22131, 783, 5548, 21177, 17253, 22207, 3056, 15321, 
	    20516, 6373, 22529, 8427, 12943, 8544, 20518, 5237, 961, 2125, 2917, 1429, 
	    15043, 19786, 8281, 4063, 8812, 753, 21791, 1469, 7778, 21829, 18545, 7100, 
	    5373, 4045, 10634, 11213, 5496, 10900, 23737, 1689, 11994, 6182, 18711, 20623, 
	    12946, 16079, 19142, 9261, 17676, 6537, 3302, 20922, 12907, 14402, 4976, 11437, 
	    19115, 8374, 13198, 22773, 13888, 15557, 22420, 23024, 20664, 1874, 19216, 
	    11839, 19994, 9563, 2021, 16417, 23760, 959, 20768, 21960, 17606, 3072, 13831, 
	    6707, 7092, 4469, 2477, 5167, 503, 1832, 6118, 18527, 24751, 23453, 17349, 
	    20889, 21500, 5219, 2468, 318, 19425, 23067, 23640, 19316, 11349, 9937, 24718, 
	    15873, 1512, 12965, 15049, 15263, 22240, 23086, 20080, 21824, 3675, 14954, 991, 
	    14507, 13570, 14208, 13704, 23939, 6287, 582, 8066, 2198, 23020, 19868, 3240, 
	    5977, 15492, 21304, 18780, 6974, 15050, 2380, 18909, 4385, 9403, 9806, 22715, 
	    2925, 673, 1158, 6962, 15860, 5474, 20068, 16557, 21746, 5248, 8836, 9519, 
	    22503, 13441, 3195, 4072, 22211, 8733, 6203, 14369, 18978, 19524, 2167, 7784, 
	    10998, 21289, 16628, 3259, 3353, 18461, 1639, 23755, 7771, 21362, 20153, 10281, 
	    21340, 6249, 313, 18217, 5366, 7797, 6520, 18835, 17452, 14924, 18839, 15065, 
	    10982, 15505, 6864, 24952, 17794, 2722, 3902, 12629, 7509, 15537, 19432, 21958, 
	    5990, 21043, 9903, 22154, 22363, 19581, 20804, 8656, 6117, 5229, 17602, 7937, 
	    10051, 17497, 16177, 20592, 12263, 10161, 6382, 17822, 12904, 16669, 16445, 
	    10101, 23087, 23619, 16979, 16080, 10961, 24199, 17011, 11790, 2675, 5943, 361, 
	    14221, 1708, 5202, 6523, 5178, 16543, 12860, 4157, 1436, 13697, 23271, 21518, 
	    11498, 19722, 8738, 6188, 2452, 21638, 15710, 18439, 12287, 1617, 10219, 24849, 
	    2262, 23704, 6721, 23595, 22440, 10168, 1126, 23100, 940, 5398, 10386, 1536, 
	    8549, 10709, 23387, 4608, 11229, 2836, 17751, 24995, 2178, 2251, 12177, 18992, 
	    7127, 15002, 9294, 6265, 24281, 1513, 23300, 21748, 15230, 19345, 17766, 5482, 
	    6927, 18093, 3895, 6137, 14850, 18310, 700, 15969, 14379, 6481, 4220, 6569, 
	    5890, 12935, 13161, 8802, 20835, 22048, 19917, 18008, 9891, 14313, 20370, 1351, 
	    3004, 22439, 10710, 7719, 6235, 4392, 9054, 4729, 22996, 12831, 7293, 13599, 
	    6273, 15656, 6291, 15968, 1083, 15066, 22338, 23534, 21766, 14818, 13124, 11121, 
	    22931, 23248, 1731, 13466, 9635, 22545, 20546, 2300, 3951, 12111, 17406, 5755, 
	    22598, 15622, 17075, 10362, 7245, 20174, 6679, 19093, 20129, 21390, 24475, 
	    23985, 10542, 9314, 8554, 21159, 11292, 2560, 15741, 7250, 22421, 5207, 19531, 
	    19343, 10220, 3877, 2435, 13841, 13926, 13371, 8606, 23890, 9138, 12912, 21810, 
	    11154, 16917, 1948, 16086, 24423, 1507, 6622, 12061, 3649, 11788, 5351, 920, 
	    3158, 2467, 18846, 14225, 11915, 24838, 2542, 20138, 994, 18045, 22061, 3402, 
	    17340, 14558, 4188, 6734, 1422, 19933, 24924, 8748, 23876, 8158, 17033, 22699, 
	    21628, 19749, 5564, 18777, 17850, 13016, 14410, 17430, 14879, 21514, 21522, 
	    12933, 1941, 22192, 7555, 8431, 14946, 15957, 13810, 9422, 16452, 12037, 13797, 
	    320, 15176, 18876, 11481, 10633, 4209, 24377, 18998, 21355, 7926, 16280, 19228, 
	    6006, 2951, 6177, 23228, 18440, 23164, 18450, 359, 2673, 13823, 11853, 18089, 
	    271, 21013, 4529, 7532, 17155, 10065, 8481, 21571, 474, 15067, 12613, 1787, 
	    15595, 24821, 4113, 10500, 11990, 987, 6608, 19462, 14747, 24002, 7979, 24917, 
	    24155, 5799, 2989, 2360, 5353, 16208, 24631, 10807, 16125, 2250, 9170, 9909, 
	    16884, 19078, 9979, 14215, 4272, 6588, 9719, 10504, 12666, 23135, 15518, 20733, 
	    20983, 7935, 13126, 15892, 799, 1143, 8453, 12874, 6061, 23711, 22777, 19321, 
	    14916, 13450, 20814, 7503, 21823, 17428, 7304, 5003, 19752, 1972, 18464, 1841, 
	    2606, 15703, 3297, 11708, 8034, 20860, 6644, 11933, 10999, 1718, 7369, 16301, 
	    11578, 9744, 17145, 18296, 3783, 5166, 19363, 228, 956, 23402, 21490, 5793, 
	    11274, 10519, 19573, 2320, 14676, 14737, 8433, 23119, 21991, 14878, 12347, 5996, 
	    14725, 17777, 23076, 2626, 5290, 20227, 19433, 9473, 18420, 6454, 2736, 15280, 
	    10173, 2226, 6641, 18568, 19721, 23626, 14892, 9579, 1889, 7678, 10905, 23356, 
	    6806, 21569, 9103, 20436, 12601, 17936, 2479, 9865, 24301, 4971, 15643, 12771, 
	    14428, 3099, 17522, 24442, 14139, 12890, 19148, 8389, 12962, 13796, 8174, 6890, 
	    21838, 15880, 18289, 4285, 18853, 24492, 20559, 4180, 21495, 3565, 15370, 21341, 
	    17131, 2501, 16463, 7627, 8616, 23775, 12672, 20497, 22465, 2211, 16227, 11352, 
	    5251, 9879, 10318, 20556, 13611, 9113, 22666, 22751, 20988, 1876, 16001, 3141, 
	    19324, 15912, 19712, 12754, 14108, 16881, 11692, 22017, 17535, 6152, 6456, 
	    12788, 6385, 8658, 20706, 16916, 682, 9003, 22435, 222, 11835, 10783, 16075, 
	    11891, 18656, 10237, 3272, 16587, 24879, 8822, 10834, 4638, 18906, 9401, 21065, 
	    17925, 13773, 24832, 9587, 14505, 6755, 14541, 7864, 4764, 2396, 21995, 18767, 
	    6068, 22528, 250, 7847, 14487, 17326, 20448, 11987, 8172, 6387, 10875, 13712, 
	    4736, 19805, 8298, 134, 18911, 11797, 1149, 9511, 2126, 18953, 2797, 10615, 984, 
	    11816, 4215, 15876, 5733, 9454, 14773, 3323, 15817, 3247, 24374, 14252, 3080, 
	    21900, 6151, 5377, 17960, 20858, 11208, 24679, 22892, 21258, 357, 8596, 24616, 
	    14282, 5444, 14794, 512, 11906, 8718, 21185, 11821, 13331, 3888, 17903, 13728, 
	    24985, 19711, 10869, 5016, 5835, 17407, 1900, 19858, 15715, 3431, 4324, 13270, 
	    20651, 6169, 24391, 9647, 10052, 36, 22448, 6384, 8101, 7760, 17932, 7491, 2766, 
	    11160, 14328, 11783, 11036, 20245, 4648, 13154, 23419, 22735, 10674, 3884, 
	    16973, 10240, 9125, 2974, 19406, 8934, 1629, 8414, 21224, 1169, 13153, 22236, 
	    18063, 12669, 15108, 24514, 22424, 851, 2094, 16453, 8954, 21854, 10442, 1646, 
	    21298, 7468, 12142, 12405, 2186, 2008, 6498, 19049, 5127, 3498, 23579, 4932, 
	    1067, 12172, 12140, 8400, 2338, 13333, 24097, 14168, 17531, 19739, 7364, 22746, 
	    19230, 20688, 22004, 6838, 8732, 17885, 18751, 16039, 17421, 15829, 9386, 23765, 
	    13020, 15692, 17710, 6450, 4914, 24368, 6196, 19313, 17591, 16163, 10736, 15891, 
	    11501, 13249, 7253, 15124, 24590, 20010, 10367, 3673, 15134, 14304, 2730, 547, 
	    15192, 21970, 9416, 1701, 4534, 12500, 13971, 24175, 16978, 16491, 22314, 9081, 
	    13177, 5247, 3076, 23207, 6479, 19033, 1738, 7446, 16272, 6729, 17416, 22993, 
	    8060, 6163, 8913, 4765, 4938, 10355, 22481, 4574, 13848, 8292, 7461, 9564, 
	    19698, 16274, 2066, 3225, 3674, 17931, 20790, 22022, 6985, 20732, 21048, 3740, 
	    5263, 10959, 16545, 21821, 1021, 9706, 23709, 7477, 20466, 18156, 12098, 4941, 
	    20013, 18627, 7052, 20910, 13364, 2648, 5790, 23514, 2900, 10668, 16257, 15785, 
	    24912, 8245, 13507, 19910, 16706, 10117, 268, 6010, 17691, 8054, 18405, 9886, 
	    3912, 19847, 5194, 2296, 17213, 21997, 11712, 11369, 12850, 9134, 10404, 21256, 
	    591, 13708, 14868, 19100, 2762, 7539, 24042, 14038, 10579, 20205, 10401, 12878, 
	    20505, 21428, 17671, 1485, 10889, 23792, 20146, 16944, 17696, 5923, 8545, 2809, 
	    6315, 20362, 15325, 16339, 16766, 1480, 21156, 23338, 21702, 16702, 2418, 4910, 
	    10547, 19726, 23529, 9763, 23918, 24169, 2446, 5279, 12937, 1030, 10616, 9352, 
	    9549, 3122, 9907, 1276, 19894, 8712, 13430, 17637, 778, 2431, 2312, 7865, 11600, 
	    1856, 20570, 18050, 7463, 14834, 1711, 24533, 6500, 20209, 18413, 6574, 19584, 
	    20601, 5973, 23266, 24234, 2719, 8237, 20626, 17503, 3755, 10268, 17382, 14025, 
	    10535, 21276, 7582, 22753, 7752, 22327, 2816, 16811, 15964, 1608, 20342, 21578, 
	    15011, 4281, 6999, 3666, 6979, 7941, 3953, 10095, 9611, 9350, 18428, 20642, 
	    17961, 4262, 18042, 4077, 5011, 12533, 9187, 23104, 6814, 5002, 20344, 14020, 
	    6651, 18111, 1012, 9757, 21344, 4865, 2672, 16783, 11585, 616, 24603, 492, 3797, 
	    3185, 18605, 4304, 23675, 11505, 8403, 883, 14589, 20800, 21839, 552, 4159, 
	    3585, 18738, 19238, 4425, 14463, 19112, 24364, 11257, 15862, 22026, 21975, 7203, 
	    2106, 23406, 6881, 22379, 703, 2568, 20210, 23692, 1046, 22114, 5560, 11280, 
	    14776, 13494, 10602, 19755, 6723, 4111, 5906, 17642, 9745, 1317, 21150, 15867, 
	    14729, 13849, 12268, 9385, 21471, 17969, 21056, 17212, 1778, 5313, 2243, 511, 
	    24015, 10084, 16441, 397, 16924, 12047, 10030, 2750, 17306, 13779, 447, 17204, 
	    2297, 12780, 21836, 8320, 21297, 13936, 10732, 4084, 9192, 14689, 22091, 19205, 
	    21919, 15821, 16153, 16707, 23722, 1520, 7132, 2646, 5584, 5070, 18857, 4918, 
	    8395, 1266, 14205, 2112, 8856, 12972, 2883, 13021, 17763, 24621, 20381, 17184, 
	    20903, 1084, 1384, 2664, 7281, 7485, 12506, 14187, 20135, 16256, 4054, 15496, 
	    15581, 15129, 24866, 21473, 24401, 9315, 4038, 3995, 17013, 18572, 14666, 13111, 
	    13987, 371, 4, 5877, 23418, 15893, 2814, 23855, 20185, 24201, 18328, 24799, 
	    21812, 20248, 24441, 11005, 20224, 19906, 11957, 15989, 6318, 24123, 13856, 
	    16855, 1685, 11108, 2409, 104, 24392, 23456, 22613, 24903, 2064, 23455, 22321, 
	    312, 15513, 21885, 19338, 20197, 15568, 9447, 12926, 14978, 7079, 10765, 11136, 
	    654, 17273, 7845, 4276, 19888, 24488, 19514, 2388, 11260, 11317, 18128, 14943, 
	    7082, 16730, 17316, 5153, 16804, 6968, 14451, 17341, 24500, 6294, 1024, 11226, 
	    14126, 14686, 8773, 12463, 7552, 6795, 10646, 5997, 11583, 20300, 10124, 22929, 
	    8269, 2792, 20400, 19653, 17098, 19493, 13806, 18531, 15127, 20606, 22840, 
	    14927, 10527, 20456, 18577, 4724, 18729, 5955, 16551, 20674, 23526, 18762, 
	    14658, 24777, 4536, 17635, 12185, 21945, 15716, 16635, 23687, 9503, 8150, 6774, 
	    11085, 2018, 4845, 19287, 14540, 17370, 13077, 19035, 22098, 24236, 19586, 6180, 
	    9274, 15971, 1309, 566, 14332, 17547, 24812, 16014, 6499, 7635, 5766, 23163, 
	    4510, 1602, 22107, 21033, 7993, 24389, 9177, 18504, 21392, 14491, 15905, 18670, 
	    6423, 696, 12191, 11185, 4368, 22168, 12807, 19140, 17632, 19920, 41, 1980, 
	    19848, 16499, 3169, 18359, 14345, 15538, 7525, 7860, 11759, 1652, 5452, 20661, 
	    8367, 18184, 24000, 21281, 7587, 1661, 9469, 21808, 13908, 19288, 20956, 18309, 
	    89, 12755, 19533, 18163, 17283, 6693, 3500, 15938, 24110, 20510, 9728, 10082, 
	    16824, 18471, 21251, 6405, 1999, 5941, 9699, 11506, 13045, 21752, 3113, 22788, 
	    7308, 8264, 14874, 19955, 16954, 9824, 1826, 9067, 23470, 24231, 16006, 2227, 
	    22182, 19282, 2930, 22378, 6760, 11903, 23121, 22960, 3102, 10402, 19461, 9045, 
	    4306, 639, 2760, 2471, 13808, 5519, 18770, 17330, 13471, 15502, 22691, 16528, 
	    6322, 20735, 10714, 21014, 17137, 17809, 13775, 11735, 4028, 23507, 1085, 4369, 
	    14174, 19026, 9075, 656, 13813, 11258, 11170, 3913, 10956, 17429, 24690, 8852, 
	    13052, 4172, 5142, 15419, 22957, 24921, 3065, 4812, 9776, 15849, 6700, 13722, 
	    3254, 21041, 19297, 22143, 21441, 478, 14717, 4030, 15921, 18872, 19819, 19046, 
	    4238, 16167, 21439, 4459, 21015, 1047, 10692, 3723, 23801, 5600, 9017, 4835, 
	    12970, 10514, 22537, 18638, 20876, 10815, 11843, 21694, 7872, 15171, 17185, 
	    5015, 19964, 20257, 2891, 3660, 11603, 3717, 22862, 9237, 10803, 10049, 1966, 
	    20450, 14093, 807, 3856, 24900, 14273, 11802, 16778, 18834, 21200, 17945, 17095, 
	    5037, 12894, 7840, 16921, 24935, 21054, 3066, 20419, 2193, 22116, 14660, 16583, 
	    23393, 18620, 15712, 5845, 2374, 13859, 136, 22992, 4927, 13442, 5345, 11852, 
	    20481, 11475, 2438, 18486, 21581, 19965, 16867, 12444, 6844, 24701, 15470, 
	    12103, 3886, 4429, 3619, 9466, 24069, 8569, 22361, 16225, 15752, 16892, 16420, 
	    24623, 23449, 21192, 14817, 7533, 1476, 3975, 3284, 674, 3914, 24103, 5882, 
	    24843, 9608, 16507, 2688, 1945, 6601, 16949, 5935, 17175, 6739, 12660, 16288, 
	    19185, 20727, 19059, 10915, 8745, 11227, 16728, 19063, 23488, 21624, 16296, 112, 
	    5462, 13202, 23889, 12203, 14745, 1662, 18072, 11049, 3505, 24137, 15811, 22953, 
	    11650, 14989, 19523, 8711, 10291, 8003, 1618, 17344, 13228, 3190, 1736, 13617, 
	    6871, 23659, 10196, 18571, 20161, 16991, 8309, 13909, 3305, 22681, 16215, 17564, 
	    247, 24337, 18803, 8015, 6470, 23278, 11532, 23895, 18761, 5272, 8857, 1634, 
	    17515, 13490, 10843, 20711, 2098, 9460, 1730, 3617, 5206, 6914, 7025, 2268, 
	    2424, 11784, 4558, 16994, 9791, 6028, 12274, 20173, 24589, 15942, 8396, 8920, 
	    4635, 13977, 12658, 9639, 18116, 993, 15720, 7544, 4787, 20766, 15883, 4640, 
	    18041, 18958, 5685, 17607, 9983, 3587, 19227, 10097, 21096, 24096, 17048, 14782, 
	    2020, 20256, 12993, 1119, 8052, 892, 23676, 3809, 13251, 16328, 6621, 2432, 
	    2347, 18228, 148, 13920, 18160, 16987, 20334, 13504, 17990, 4936, 23552, 19669, 
	    5195, 15868, 19404, 11458, 10952, 1195, 24885, 20564, 10420, 393, 19715, 23141, 
	    9915, 12497, 22934, 22616, 15780, 15298, 17624, 23, 2218, 7698, 22112, 14662, 
	    1748, 24716, 9898, 3149, 6867, 11393, 15549, 8854, 1112, 20449, 10808, 2610, 
	    2823, 20701, 12733, 97, 10515, 9018, 18215, 6186, 20151, 5384, 1417, 18786, 
	    9672, 670, 2249, 21523, 2173, 6269, 549, 13825, 15521, 986, 23859, 1105, 18339, 
	    1427, 17459, 8679, 23714, 144, 9546, 10421, 22731, 6229, 18787, 23360, 11044, 
	    16343, 17883, 10718, 22560, 15144, 6506, 17754, 12900, 2045, 17827, 16690, 
	    14925, 4564, 14613, 1129, 21545, 5664, 10620, 16124, 22044, 18452, 21655, 381, 
	    253, 20038, 8881, 14635, 1048, 8673, 17270, 5329, 23804, 12454, 3665, 14957, 
	    12699, 16682, 22494, 9027, 21769, 7424, 20947, 9275, 13132, 23749, 11594, 24904, 
	    21418, 24662, 8067, 16525, 11125, 23621, 6923, 23695, 17127, 20932, 330, 6610, 
	    15548, 16146, 2611, 1969, 8703, 18808, 6383, 1339, 15585, 123, 16110, 23003, 
	    2181, 7318, 2323, 17263, 10590, 4785, 8345, 19226, 2164, 23847, 24091, 8070, 
	    262, 7536, 17887, 15101, 17576, 21494, 17004, 21012, 16365, 12732, 6320, 18415, 
	    24305, 1939, 2047, 11413, 18404, 6351, 15542, 5700, 1991, 9896, 21364, 5739, 
	    16381, 12677, 6505, 21737, 23061, 22427, 8879, 395, 22178, 4786, 8413, 19925, 
	    2328, 820, 8520, 10882, 12473, 21375, 3037, 10890, 22631, 6088, 4473, 14855, 
	    1951, 6433, 15149, 18993, 13085, 24287, 16161, 17654, 24203, 1410, 19779, 18374, 
	    18594, 4830, 11350, 18164, 5231, 8319, 7776, 23800, 10294, 19995, 15743, 5804, 
	    6344, 2575, 22261, 23682, 4380, 8020, 4384, 19333, 16934, 7733, 23997, 5157, 
	    21368, 8461, 10027, 10981, 11456, 15992, 9012, 13576, 12823, 3165, 1953, 7076, 
	    1740, 21489, 7290, 7798, 6367, 2054, 13176, 18059, 1125, 17404, 14015, 14827, 
	    19362, 20420, 734, 18633, 15693, 22671, 2829, 5781, 15990, 24072, 20123, 11175, 
	    14314, 7456, 9875, 20854, 11422, 9514, 22348, 15165, 4544, 6089, 16170, 20612, 
	    17352, 969, 10387, 20942, 5572, 15345, 23684, 24817, 18125, 21000, 22212, 15749, 
	    23900, 4679, 16098, 23909, 14253, 8684, 5344, 18119, 2738, 24243, 15036, 15229, 
	    23849, 16540, 2044, 5663, 18868, 9691, 6846, 5169, 2401, 8317, 3456, 6528, 
	    23296, 548, 4377, 13655, 1809, 2398, 18435, 13424, 18691, 9496, 5414, 21866, 
	    24445, 22714, 4586, 16260, 6199, 21996, 4128, 37, 7173, 22589, 1053, 9797, 266, 
	    21453, 10433, 887, 13483, 10921, 15563, 24820, 8782, 20830, 19855, 15851, 10410, 
	    12990, 24991, 7732, 19684, 11455, 9949, 11210, 22897, 17456, 7657, 22075, 9226, 
	    15248, 19747, 8531, 14386, 17852, 15344, 9652, 16266, 20613, 1601, 14964, 4480, 
	    11323, 1967, 21313, 2428, 5597, 18179, 423, 21091, 19763, 5240, 9364, 8959, 
	    20721, 17954, 21515, 17842, 23721, 7989, 20939, 21075, 8823, 14503, 22133, 
	    19733, 22557, 3300, 3325, 1359, 23380, 14394, 10576, 13754, 11179, 15659, 23093, 
	    18651, 18615, 18641, 19129, 14917, 7405, 23008, 15387, 3150, 11314, 4401, 16246, 
	    6114, 3892, 16639, 21312, 4666, 22204, 6929, 13141, 2680, 17630, 16990, 19285, 
	    18122, 17757, 2212, 9505, 6189, 10700, 6301, 15466, 16895, 5450, 9457, 16533, 
	    6710, 1457, 23186, 6638, 11717, 21585, 1827, 4554, 24272, 24165, 7256, 22566, 
	    13522, 18515, 19724, 17742, 11230, 3281, 23209, 10194, 12706, 12720, 23353, 
	    6842, 8363, 23601, 9049, 7144, 3464, 20607, 16744, 24178, 1908, 793, 21024, 
	    21681, 11500, 13172, 6483, 4516, 21834, 6284, 8890, 4563, 14694, 707, 7852, 
	    10786, 1407, 551, 24521, 13902, 24760, 3168, 15162, 20912, 6663, 23651, 18090, 
	    14319, 14496, 6263, 1353, 19662, 20485, 21006, 20476, 15823, 432, 12536, 15003, 
	    19061, 7543, 17125, 4647, 14323, 24975, 4669, 7918, 9242, 3872, 1418, 2281, 
	    23541, 5917, 4654, 13714, 15979, 1839, 6014, 8372, 12576, 4580, 9657, 22745, 
	    16505, 6642, 6431, 14006, 16196, 12081, 21990, 9827, 5902, 16532, 8513, 13090, 
	    7158, 11897, 24801, 17760, 3471, 11078, 6400, 6390, 15299, 13118, 12182, 943, 
	    15564, 18675, 11439, 16254, 15015, 20827, 21614, 2402, 18488, 9696, 23783, 
	    22489, 8100, 6063, 13093, 14579, 24477, 19767, 24588, 12885, 6403, 22170, 1379, 
	    23211, 14911, 16845, 12748, 6371, 5847, 17634, 12176, 12820, 17750, 19997, 9882, 
	    7557, 9459, 1903, 24792, 8143, 9762, 3057, 1663, 13938, 19614, 4120, 10823, 
	    20297, 17546, 8165, 7851, 2288, 10503, 24304, 6765, 17942, 19106, 14807, 12000, 
	    16968, 1946, 4086, 47, 646, 11140, 10516, 13601, 2229, 23567, 11813, 17481, 
	    7217, 8581, 3624, 9428, 15153, 17892, 23597, 21465, 24897, 2609, 20553, 10341, 
	    1630, 7498, 13357, 13582, 4019, 20777, 17226, 9160, 11740, 4485, 4303, 18575, 
	    1845, 12252, 12539, 17369, 14797, 23685, 19511, 19598, 15913, 8666, 24554, 162, 
	    1609, 10712, 8189, 9834, 18706, 23647, 17904, 24028, 11993, 7996, 3380, 13783, 
	    19023, 1944, 5875, 19372, 4174, 11305, 22037, 22935, 260, 10113, 14646, 21360, 
	    16776, 3829, 16426, 6408, 15747, 11022, 1892, 906, 13394, 7613, 5516, 15175, 
	    21859, 12313, 22864, 15089, 24065, 5064, 21507, 7529, 5761, 20852, 4034, 4542, 
	    12731, 23554, 15267, 21127, 10282, 4237, 8257, 17237, 1642, 6496, 24629, 22762, 
	    14432, 20781, 7270, 23542, 17914, 15357, 6568, 12331, 23852, 22275, 17977, 337, 
	    4937, 3293, 13002, 4531, 17337, 12006, 2570, 1349, 8598, 10937, 10016, 3926, 
	    8946, 12843, 9083, 9433, 8356, 5455, 23090, 4779, 23962, 14546, 15333, 24336, 
	    2656, 7957, 17418, 17813, 3013, 20051, 23448, 527, 5521, 8951, 19273, 24587, 
	    14983, 409, 24964, 19485, 15107, 9424, 8610, 17899, 13524, 19551, 1449, 10314, 
	    20016, 21825, 9663, 24766, 9637, 22232, 17709, 5767, 9383, 7925, 9782, 18195, 
	    15838, 22083, 6364, 115, 20131, 17880, 20773, 11331, 11336, 20900, 12782, 14287, 
	    21631, 13411, 2259, 13381, 18302, 16115, 15771, 1162, 2473, 4639, 20063, 10543, 
	    9575, 22350, 2042, 3712, 3497, 15389, 19543, 11560, 5076, 16443, 5614, 24894, 
	    1977, 19051, 21927, 15567, 8503, 15279, 1009, 957, 20392, 10513, 7178, 9944, 
	    13766, 20250, 12887, 10147, 9595, 15337, 16943, 21599, 24637, 22059, 19088, 
	    24656, 629, 12526, 11288, 16469, 13150, 19579, 22331, 10541, 13861, 21132, 
	    23650, 3765, 6589, 838, 3274, 2638, 3950, 23770, 2373, 10245, 22583, 22337, 
	    6922, 7680, 11750, 20708, 12058, 23151, 872, 2990, 11152, 20414, 4951, 13297, 
	    11483, 3242, 14986, 21029, 1569, 868, 7648, 2825, 14327, 5779, 17707, 22329, 
	    6775, 16521, 6370, 11358, 19449, 15377, 9656, 17276, 9429, 1567, 8948, 7599, 
	    8868, 10934, 18355, 11620, 14886, 23412, 11732, 18234, 16474, 2403, 2482, 23686, 
	    21098, 19267, 5513, 20233, 5859, 13993, 14992, 19122, 7563, 22187, 6074, 1163, 
	    21876, 3859, 9833, 12855, 16501, 19599, 18271, 17314, 6873, 18541, 2747, 17361, 
	    10852, 8045, 19155, 24387, 6170, 23101, 10205, 12398, 23317, 17700, 7249, 21629, 
	    4444, 8853, 10909, 1020, 1554, 4286, 1345, 14509, 960, 22531, 21531, 24081, 
	    16344, 941, 5339, 3900, 5236, 10530, 3756, 3870, 14788, 12238, 16519, 12808, 
	    8246, 12975, 19534, 5734, 12955, 17815, 19594, 9980, 10409, 7703, 5796, 18053, 
	    4345, 19641, 12101, 16399, 16958, 6154, 14207, 16880, 18330, 13359, 759, 488, 
	    21393, 19440, 24370, 18895, 22853, 23227, 18400, 11937, 3813, 577, 15523, 8124, 
	    20816, 10655, 22406, 8995, 14290, 1154, 19537, 15897, 4535, 10752, 14733, 13749, 
	    3589, 24694, 18005, 12803, 7233, 1879, 21267, 13684, 2363, 21327, 7893, 1932, 
	    13224, 15510, 5562, 506, 4799, 20737, 17367, 21282, 254, 11913, 636, 5141, 
	    23458, 6835, 7266, 23011, 22051, 19032, 5348, 7162, 12616, 2803, 24816, 10184, 
	    13217, 2663, 2758, 17868, 16625, 12832, 3231, 11888, 22547, 6686, 17866, 8918, 
	    35, 20891, 17169, 15346, 8780, 23482, 13560, 21002, 1695, 6101, 20502, 5702, 
	    5803, 909, 1543, 15973, 12543, 24568, 10305, 10988, 3852, 5594, 8028, 4057, 
	    13600, 14853, 4047, 2608, 7125, 11693, 837, 7112, 22077, 23429, 2357, 15565, 
	    6821, 17851, 22536, 11762, 2453, 2305, 19902, 13322, 6910, 11765, 14027, 10725, 
	    5510, 8614, 1006, 3090, 6880, 22643, 21939, 14999, 15412, 13095, 6658, 10358, 
	    14996, 10864, 13744, 7055, 9530, 23832, 3123, 4616, 13326, 5820, 4694, 22988, 
	    23907, 6945, 4278, 6050, 12115, 5280, 6478, 8069, 18388, 22543, 12768, 18904, 
	    1253, 15809, 11354, 15946, 20348, 10779, 23423, 1721, 5389, 11611, 20462, 14154, 
	    1982, 9406, 5711, 10995, 9047, 22965, 4678, 13833, 14519, 6245, 7356, 3194, 
	    13905, 3987, 23878, 21026, 5282, 6005, 21440, 3114, 18866, 22385, 23339, 8004, 
	    12925, 24592, 4789, 15110, 7829, 1146, 13315, 23641, 7284, 7141, 5458, 12007, 
	    14640, 15926, 16089, 18703, 8196, 4387, 4001, 3018, 824, 6398, 15944, 21938, 
	    224, 2607, 23383, 18844, 24988, 10417, 16055, 22014, 8636, 18292, 6261, 18873, 
	    1172, 294, 20615, 1055, 13652, 9267, 3248, 13638, 4855, 897, 22622, 692, 4915, 
	    15759, 12643, 404, 2393, 7328, 13961, 19921, 12204, 12013, 9298, 7506, 15030, 
	    7530, 2859, 7090, 12738, 19723, 22149, 10414, 19638, 12438, 15805, 4078, 19498, 
	    10619, 8463, 6858, 24889, 20390, 11836, 8213, 14086, 4637, 14875, 21358, 10505, 
	    10422, 17594, 19280, 22266, 19464, 17625, 11337, 5780, 22986, 23857, 18833, 
	    2309, 12258, 13160, 2127, 9987, 18006, 3516, 14730, 7238, 21165, 19575, 14091, 
	    475, 14284, 2141, 8411, 4940, 24702, 3580, 2425, 1124, 1242, 4772, 12983, 16105, 
	    3384, 18047, 3138, 20736, 11289, 2498, 12546, 21257, 4559, 20578, 20007, 21855, 
	    11299, 5051, 6237, 12426, 8728, 18565, 6681, 15553, 20632, 15086, 12610, 9845, 
	    16336, 7749, 15701, 3828, 24615, 24673, 17894, 3925, 13062, 12560, 8622, 1322, 
	    17188, 18499, 8818, 3159, 7887, 10335, 13703, 12084, 2892, 12242, 13038, 3294, 
	    23668, 7622, 14831, 6892, 603, 6628, 15981, 5691, 11965, 10724, 15948, 5867, 
	    8695, 24759, 23473, 16168, 7041, 19945, 15269, 18563, 5568, 420, 11688, 11530, 
	    18174, 3651, 12949, 15836, 16340, 8167, 10979, 24956, 12193, 5992, 10202, 1388, 
	    12439, 24611, 6218, 24948, 17025, 17527, 8016, 13574, 22502, 835, 4884, 14761, 
	    17486, 3654, 14681, 14751, 841, 19330, 18232, 1844, 15915, 10492, 21577, 7583, 
	    1502, 11461, 13729, 380, 15552, 16588, 8831, 13640, 13413, 17703, 24689, 13532, 
	    2623, 16101, 6347, 8643, 17832, 12074, 13098, 10790, 8135, 11588, 12136, 20062, 
	    20605, 16023, 8592, 18975, 10493, 21738, 2209, 23798, 760, 17493, 16731, 13216, 
	    23772, 12621, 17759, 18957, 8587, 24989, 8905, 15744, 3366, 9747, 6714, 716, 
	    19646, 14575, 5493, 2349, 12323, 3124, 18482, 20073, 20646, 10509, 10625, 15535, 
	    9726, 3650, 23274, 5666, 11407, 11960, 10457, 9590, 18146, 9101, 24560, 7181, 
	    17570, 8362, 16493, 9039, 23036, 6667, 21686, 14869, 3264, 5899, 3555, 16781, 
	    21268, 23802, 10839, 5945, 15452, 4683, 13258, 11371, 8574, 7471, 17138, 21669, 
	    23701, 8709, 14435, 15384, 21318, 13432, 8808, 9348, 19199, 12249, 7466, 21491, 
	    9621, 19224, 3504, 6429, 5041, 1405, 3569, 1653, 1275, 20787, 6285, 24113, 9483, 
	    9703, 21339, 24326, 15353, 607, 4550, 7634, 18099, 17715, 18532, 12303, 15636, 
	    1265, 10689, 10413, 4805, 1896, 14727, 23921, 13089, 18108, 19138, 15088, 24877, 
	    12512, 13727, 7814, 9623, 8547, 12707, 945, 8760, 18407, 23241, 7189, 9764, 
	    3151, 6773, 19529, 5115, 23032, 8876, 22936, 22078, 14360, 24050, 14114, 3267, 
	    6630, 7917, 18196, 19624, 17444, 8862, 19613, 11634, 2304, 10648, 17733, 8525, 
	    21254, 5277, 10904, 19567, 23531, 17758, 14795, 3212, 16673, 686, 5036, 24261, 
	    3232, 5769, 8133, 22405, 14354, 18383, 12619, 7330, 608, 8795, 8455, 20160, 
	    17718, 18845, 23219, 15982, 12154, 18977, 14711, 18142, 5927, 24142, 14268, 
	    5529, 16410, 22262, 917, 5726, 6828, 17686, 7646, 7684, 6730, 9475, 11971, 
	    23901, 14641, 21911, 16150, 1928, 21046, 11705, 2264, 24712, 20474, 16120, 
	    12913, 24566, 11938, 22339, 10471, 6158, 15516, 155, 15376, 18669, 2887, 10434, 
	    11328, 19568, 9043, 2517, 12545, 6761, 15519, 15094, 13058, 17053, 7656, 12586, 
	    23475, 8984, 1913, 11940, 10482, 1883, 17688, 16876, 1437, 17491, 16582, 6029, 
	    6411, 14990, 18166, 23355, 9718, 22776, 9146, 24269, 10251, 20864, 6627, 23043, 
	    13163, 22179, 12234, 22023, 15138, 19674, 660, 16735, 22742, 11040, 20633, 
	    22310, 2069, 20204, 4479, 24744, 2009, 1439, 17873, 13087, 9686, 7309, 17220, 
	    14132, 4044, 17451, 21121, 13604, 23367, 18743, 3662, 15637, 12071, 3346, 21032, 
	    6356, 17425, 612, 10770, 12218, 11876, 20008, 14560, 3419, 17225, 22843, 20130, 
	    11271, 8428, 3858, 8375, 17803, 21174, 2434, 7512, 24177, 11537, 2207, 14028, 
	    24974, 3599, 7457, 12262, 9793, 16959, 22530, 17311, 21172, 5938, 9270, 4349, 
	    23924, 18805, 20749, 23784, 19083, 13474, 16155, 14002, 17097, 14217, 22206, 
	    2998, 20021, 20934, 17047, 15795, 14112, 11894, 17141, 8891, 13263, 11616, 
	    16211, 938, 3613, 5370, 18139, 7261, 24068, 5062, 13536, 5555, 15697, 23447, 
	    6124, 14606, 12495, 1681, 23740, 19967, 17575, 6131, 17729, 3172, 19991, 16305, 
	    19190, 1250, 7883, 16043, 7451, 18890, 7274, 3930, 7788, 4810, 24677, 13425, 
	    16763, 4095, 16113, 1765, 13076, 12923, 24340, 6852, 8628, 7639, 7272, 21632, 
	    10844, 20098, 14471, 864, 9472, 11796, 18429, 12195, 17032, 22471, 12415, 2111, 
	    1225, 24321, 4624, 24916, 8212, 5284, 20815, 12178, 2811, 15635, 13740, 23078, 
	    15584, 20041, 20884, 11646, 17727, 18312, 13858, 7161, 14535, 18987, 21157, 
	    24025, 21551, 3753, 4080, 23799, 5312, 5620, 18496, 17108, 8404, 24871, 16516, 
	    13999, 9313, 8125, 24941, 21249, 22820, 23680, 15363, 7691, 11895, 10750, 2720, 
	    17830, 15480, 19817, 4870, 22233, 1535, 16935, 16717, 18016, 24909, 13954, 
	    19247, 2653, 8716, 21754, 16154, 5692, 10029, 9308, 1688, 20017, 5840, 6684, 
	    21310, 23563, 15278, 8122, 5235, 7070, 20029, 146, 5252, 1973, 5048, 1829, 
	    14602, 2148, 3236, 19652, 20744, 20617, 6626, 19889, 23198, 7638, 559, 3035, 
	    11166, 14577, 19942, 14889, 17470, 16936, 4097, 5826, 24986, 17066, 2131, 9811, 
	    17059, 4527, 19208, 17841, 16827, 11495, 22007, 18406, 20113, 2545, 22010, 7687, 
	    14259, 6529, 7659, 6535, 2204, 13048, 20943, 24715, 6647, 7428, 13631, 12335, 
	    6986, 2822, 4713, 7682, 18720, 1106, 22945, 13959, 18692, 11967, 23723, 6677, 
	    1064, 9399, 9220, 2155, 13629, 3876, 19053, 9310, 8005, 17221, 8103, 15641, 
	    13572, 19764, 21118, 24674, 17895, 3958, 24517, 13683, 10178, 4176, 21661, 9174, 
	    22576, 6463, 13418, 9222, 17374, 3971, 13635, 19515, 15115, 10945, 15302, 3175, 
	    12025, 19165, 22910, 17442, 20157, 6966, 8602, 11471, 24129, 6448, 24047, 9066, 
	    3028, 9418, 12844, 24411, 15438, 19542, 13682, 163, 22793, 10948, 12953, 12595, 
	    18225, 17938, 10785, 998, 5669, 15051, 19217, 20333, 5456, 12286, 948, 4460, 
	    17109, 20294, 18884, 19562, 16026, 18501, 11920, 3408, 12477, 1343, 18928, 
	    16857, 23628, 17902, 14288, 24341, 20405, 17507, 22937, 7965, 19272, 10289, 
	    9985, 1019, 13457, 8306, 725, 23537, 18068, 2253, 10973, 21984, 2997, 17958, 
	    8778, 13887, 8191, 15474, 13436, 19310, 12132, 18044, 14110, 17731, 22514, 
	    17089, 16604, 12628, 1532, 22653, 24668, 17792, 4770, 19296, 16918, 13980, 
	    17862, 16602, 14123, 1818, 19795, 13482, 22683, 13747, 16077, 11453, 6611, 2696, 
	    10060, 5693, 7625, 10577, 20933, 1743, 14680, 22733, 11638, 8096, 18315, 13540, 
	    22593, 23311, 7183, 8762, 17036, 15407, 3371, 10000, 6475, 13739, 11339, 6624, 
	    2912, 1806, 1637, 22386, 11614, 5501, 16368, 1156, 316, 7730, 6983, 4887, 22447, 
	    15141, 780, 15481, 1986, 11582, 16814, 16485, 7171, 18583, 18056, 16664, 4948, 
	    24748, 398, 13186, 23967, 3726, 6580, 17405, 20680, 13167, 7115, 21018, 5707, 
	    13260, 1746, 23229, 20807, 4035, 18397, 21693, 2888, 24524, 15768, 9259, 5332, 
	    9342, 19476, 20203, 2950, 20695, 2924, 3551, 22848, 171, 13157, 4893, 2015, 
	    5912, 14173, 12376, 13000, 8519, 3928, 22905, 21928, 272, 9856, 17181, 7951, 
	    12401, 19056, 6493, 7338, 4147, 4602, 15605, 14552, 4098, 1527, 103, 14935, 
	    16966, 2216, 23140, 3391, 22685, 9991, 2982, 7426, 4310, 12461, 4977, 15241, 
	    20585, 14756, 12138, 23205, 20926, 15591, 14772, 10304, 24312, 11221, 2451, 
	    1453, 12703, 7194, 13634, 7855, 7858, 20167, 7073, 19416, 11029, 18345, 21152, 
	    16940, 22517, 4801, 24851, 6970, 24541, 21475, 22099, 7713, 21162, 12291, 13690, 
	    4892, 7340, 22710, 18836, 20525, 18190, 4277, 462, 10071, 8690, 370, 12514, 
	    14465, 3906, 10855, 15457, 12730, 19903, 6251, 22507, 8036, 12116, 16435, 12549, 
	    23646, 319, 10005, 1366, 6655, 15779, 22674, 23055, 532, 9522, 17023, 20120, 
	    3458, 15813, 10489, 17090, 1056, 19465, 10976, 342, 651, 14180, 2764, 20323, 
	    19692, 3166, 6551, 11470, 4874, 19043, 24979, 16714, 4250, 7604, 10849, 4214, 
	    15226, 23471, 3719, 13434, 13374, 2879, 15515, 2261, 7347, 1446, 7517, 14258, 
	    13732, 17829, 12199, 12186, 5904, 6099, 24092, 2345, 4436, 22047, 17227, 4217, 
	    20689, 17345, 23656, 8176, 14191, 350, 24017, 3176, 22617, 22218, 12642, 11204, 
	    23361, 20037, 12213, 11597, 4775, 21450, 10763, 4521, 12265, 22319, 17334, 
	    10938, 918, 2085, 14264, 23994, 14254, 1461, 13489, 7561, 9650, 8648, 23368, 
	    10474, 3970, 5557, 17315, 2729, 21214, 12240, 24195, 5175, 16558, 13066, 2073, 
	    13798, 11850, 23191, 14245, 12018, 16034, 20882, 5343, 21467, 1562, 21474, 
	    15082, 7476, 5590, 11364, 20418, 17529, 4122, 24242, 1004, 12815, 11081, 2269, 
	    5722, 19738, 4750, 6, 18640, 530, 6792, 7930, 1235, 9938, 77, 14832, 15434, 
	    21708, 5017, 1374, 20070, 23282, 19783, 13129, 24669, 19814, 17930, 7333, 5120, 
	    15442, 85, 13587, 21869, 10279, 21109, 4752, 5885, 24574, 8588, 11552, 16396, 
	    24553, 8162, 20849, 17135, 11171, 3342, 10290, 21379, 15907, 17328, 8700, 4933, 
	    21038, 4546, 2074, 11368, 21802, 6674, 12727, 14090, 19071, 18392, 3275, 9002, 
	    10398, 18870, 7275, 13832, 16580, 12893, 12048, 6507, 16072, 15460, 12579, 
	    16942, 7480, 23603, 23346, 9565, 11949, 6204, 2213, 12012, 24153, 9319, 13661, 
	    14977, 20454, 5748, 10517, 15764, 19221, 20899, 3706, 14176, 3609, 2119, 7712, 
	    6012, 21107, 10039, 4365, 17513, 17154, 17779, 21705, 19753, 7743, 12670, 14759, 
	    4676, 19222, 5316, 5978, 17230, 7222, 16535, 14667, 14675, 21389, 21082, 3164, 
	    22847, 17395, 24872, 24802, 23825, 15676, 2507, 7677, 22568, 23794, 6415, 4229, 
	    10134, 2406, 8607, 1676, 14186, 12097, 8357, 13789, 4606, 20611, 9032, 13922, 
	    4690, 15106, 15639, 1426, 6797, 5028, 5866, 20099, 20071, 6757, 14246, 989, 
	    10675, 8860, 6746, 9304, 19621, 9374, 189, 4642, 79, 20101, 9395, 24610, 14417, 
	    20846, 13227, 4279, 8468, 24887, 11122, 8142, 15163, 17291, 20295, 21653, 16345, 
	    4822, 23410, 23267, 23787, 10955, 1197, 23010, 3368, 19080, 20027, 20511, 2844, 
	    9863, 14437, 22432, 1109, 17859, 24449, 23689, 24657, 8909, 21840, 11412, 13312, 
	    19849, 1905, 14932, 10886, 14791, 13114, 6092, 7707, 13836, 7350, 16049, 6000, 
	    20713, 17464, 16036, 18534, 11707, 3909, 5504, 24513, 17915, 22158, 16910, 9276, 
	    3991, 779, 15728, 17060, 1899, 15782, 18893, 16779, 8046, 2694, 20132, 18474, 
	    20446, 9684, 17554, 7632, 13950, 8241, 12390, 22322, 24019, 20000, 14661, 22094, 
	    9832, 7621, 19291, 386, 9777, 553, 9205, 20841, 20015, 21898, 1587, 11824, 7215, 
	    12290, 21250, 16536, 22669, 20569, 1360, 14447, 10424, 611, 20222, 13802, 11950, 
	    13880, 1364, 17955, 4489, 160, 15726, 615, 19478, 4804, 4793, 3554, 22563, 
	    12474, 8942, 8893, 18223, 7204, 16140, 23320, 23910, 3025, 1592, 10558, 12901, 
	    24292, 10403, 9960, 19193, 10908, 23190, 12184, 11023, 13024, 15354, 9375, 
	    16948, 7568, 13305, 15355, 8159, 22423, 24811, 23663, 1716, 19782, 11939, 12208, 
	    1099, 1700, 10704, 19196, 10058, 6820, 7546, 19195, 13452, 19290, 11462, 2062, 
	    13427, 11228, 22377, 14671, 18752, 9766, 2389, 1290, 16490, 8883, 6869, 20917, 
	    3467, 5704, 496, 14247, 16654, 15707, 2319, 11425, 8746, 18526, 9515, 246, 
	    18165, 19785, 21798, 10968, 11448, 24114, 9687, 20347, 17350, 17285, 22292, 
	    23883, 5372, 13818, 5838, 11197, 14839, 10639, 15350, 10601, 10622, 10247, 
	    13485, 15778, 15504, 23401, 15482, 2979, 20936, 19667, 23270, 4762, 15508, 
	    17712, 24317, 21188, 15440, 9190, 3694, 17293, 23286, 972, 12161, 714, 5893, 
	    20458, 13931, 8161, 18736, 6671, 12710, 13589, 21741, 8927, 3568, 3750, 22425, 
	    20441, 567, 20512, 10534, 11272, 24349, 14738, 20398, 22549, 17104, 10272, 7391, 
	    22810, 15831, 22829, 5403, 15877, 11785, 2236, 885, 12288, 12746, 314, 19210, 
	    10697, 15035, 8794, 19673, 3322, 9240, 8260, 12277, 17262, 5461, 1327, 13694, 
	    3834, 2429, 17533, 13030, 20825, 15799, 6898, 6044, 17665, 15675, 12008, 953, 
	    11100, 10994, 11159, 16107, 16873, 17858, 14679, 14111, 19388, 24450, 15610, 
	    19198, 19546, 18592, 19591, 2573, 6900, 9127, 7958, 7207, 12072, 5539, 6054, 
	    12087, 22908, 4677, 22704, 18469, 19437, 1734, 16200, 23485, 933, 22455, 21630, 
	    18600, 13215, 10559, 23969, 17783, 22896, 6851, 3521, 20078, 12371, 1792, 7792, 
	    12661, 12424, 14699, 16791, 728, 13248, 13032, 4594, 2795, 13759, 11747, 5538, 
	    4509, 4725, 20144, 24864, 10317, 18335, 13707, 3470, 5019, 23635, 15526, 4024, 
	    9544, 14523, 13437, 570, 24359, 4994, 23594, 18426, 18011, 24945, 24697, 2527, 
	    1400, 9292, 2422, 16158, 8719, 1683, 10966, 3137, 16267, 14278, 20842, 12014, 
	    7948, 8499, 15633, 8458, 19312, 4407, 1636, 3518, 12741, 13803, 4221, 24171, 
	    13329, 9317, 17544, 11330, 14297, 19983, 22990, 3216, 15289, 17514, 24837, 690, 
	    18497, 4196, 7652, 12228, 16262, 15627, 22354, 12806, 542, 13454, 5137, 18226, 
	    2392, 12634, 6363, 24172, 18386, 4691, 3290, 14971, 16736, 1090, 14562, 9033, 
	    5239, 3115, 1273, 19154, 9538, 23030, 9370, 13561, 19119, 19544, 7302, 5387, 
	    12520, 24639, 19397, 9701, 4374, 9376, 23666, 23354, 15319, 456, 17281, 235, 
	    13119, 3417, 3972, 23112, 8925, 23159, 11581, 9821, 12866, 3038, 13004, 3485, 
	    122, 23251, 7053, 11446, 12639, 1312, 18303, 1236, 23019, 22767, 13687, 8956, 
	    15647, 10312, 182, 10278, 24847, 23920, 3976, 16228, 16739, 14331, 5394, 10394, 
	    16999, 22347, 18642, 20455, 1978, 16667, 17063, 8583, 17172, 14598, 17454, 3990, 
	    344, 7700, 24166, 5347, 12090, 4939, 1833, 5668, 22600, 1385, 18553, 21924, 649, 
	    20247, 6741, 877, 22863, 23536, 6311, 4254, 1956, 12083, 669, 8432, 19124, 
	    23233, 26, 13280, 15211, 6494, 14537, 20291, 5152, 21962, 2124, 929, 20134, 769, 
	    1898, 24822, 20056, 14851, 5741, 1101, 19744, 10057, 13502, 974, 9432, 19303, 
	    12614, 13639, 5517, 13516, 16214, 4567, 3448, 24218, 3836, 10657, 22972, 10308, 
	    1086, 219, 16723, 20265, 19068, 14311, 18677, 947, 16933, 3125, 5288, 24168, 
	    21729, 2800, 16479, 14309, 23446, 13663, 4042, 23796, 10465, 11545, 3678, 15924, 
	    17085, 384, 5341, 895, 10449, 10126, 13296, 14746, 2206, 18737, 21099, 5429, 
	    3427, 1372, 1383, 20725, 10043, 22899, 18863, 2341, 20379, 10540, 16972, 9674, 
	    12886, 21290, 6566, 13763, 10960, 16596, 21882, 17192, 18351, 10745, 13081, 
	    16436, 17582, 3968, 22701, 16757, 21790, 1058, 3109, 4517, 2395, 18274, 15178, 
	    12039, 3107, 23850, 11182, 12052, 7609, 24556, 21182, 9069, 22553, 24480, 18613, 
	    13272, 1583, 20994, 3850, 17093, 24291, 13302, 12951, 730, 20576, 21596, 19742, 
	    1068, 22709, 9193, 23244, 11931, 24593, 3251, 18224, 22053, 8848, 7896, 18710, 
	    18643, 2366, 9046, 5390, 1741, 7679, 7005, 1066, 811, 18551, 1342, 7894, 10653, 
	    10066, 3221, 10996, 894, 24498, 1272, 16858, 15105, 1199, 15770, 7637, 17789, 
	    5102, 2564, 17490, 23936, 16074, 4854, 2358, 22582, 1573, 8973, 13864, 20982, 
	    17943, 10687, 3083, 13241, 14781, 5887, 22459, 4137, 20499, 774, 2643, 7198, 
	    15888, 10212, 2128, 22922, 4447, 23333, 19690, 20890, 8662, 18546, 2004, 18795, 
	    13614, 21077, 14585, 4269, 17448, 9019, 9076, 2270, 14515, 24247, 15878, 8238, 
	    13492, 13205, 8018, 24191, 16081, 21086, 194, 22486, 9082, 7542, 14991, 5851, 
	    11418, 18806, 11661, 23114, 875, 23862, 3493, 9754, 4744, 11408, 372, 6302, 
	    8514, 5966, 7243, 5125, 19504, 22351, 8050, 15816, 17901, 7597, 924, 11883, 
	    19993, 648, 1308, 9884, 20974, 9517, 20624, 8148, 11722, 9142, 4388, 22126, 
	    11487, 24134, 24185, 2039, 21734, 13758, 555, 18507, 17332, 2154, 24525, 6657, 
	    621, 8284, 6231, 11133, 4548, 12797, 5641, 2665, 21770, 11930, 12064, 21594, 
	    21199, 62, 4628, 14873, 13105, 11932, 22416, 9709, 22020, 18245, 8190, 10707, 
	    23134, 2973, 19797, 6720, 3680, 6625, 1337, 9593, 1625, 11674, 19240, 20507, 
	    24458, 6268, 5653, 3898, 2220, 9380, 9512, 13781, 11988, 5833, 14742, 9336, 
	    19554, 12080, 6354, 11889, 20359, 12796, 424, 8657, 24315, 11879, 14884, 7381, 
	    15630, 23624, 16844, 13916, 9930, 1563, 21609, 24280, 24824, 5362, 14654, 2235, 
	    3063, 3100, 12910, 16046, 6127, 18923, 3087, 24754, 16270, 21343, 18959, 8685, 
	    12856, 20171, 16579, 22877, 13967, 13212, 24382, 19928, 11053, 21539, 5181, 442, 
	    14344, 16655, 3806, 23655, 6486, 2483, 1863, 8819, 19950, 14408, 145, 17360, 
	    15906, 11572, 24757, 7136, 6724, 14157, 2400, 24018, 5012, 24895, 8988, 18712, 
	    18477, 19206, 9120, 4821, 4397, 23358, 1516, 11491, 7842, 4495, 23293, 19789, 
	    5959, 10598, 9020, 24278, 9356, 6258, 2590, 9464, 12977, 2407, 19328, 3875, 
	    3062, 13951, 958, 8494, 19001, 2538, 17519, 1724, 7003, 6076, 24746, 11815, 
	    19885, 17982, 24733, 10164, 809, 13369, 1902, 16450, 12588, 12779, 7006, 3085, 
	    20928, 10595, 20803, 8982, 7317, 9948, 16293, 9692, 17301, 22819, 15014, 1330, 
	    14926, 18580, 20509, 16255, 2273, 14594, 17726, 16718, 9855, 24038, 14739, 
	    13278, 12361, 20212, 16419, 7294, 13557, 14514, 12996, 17478, 14638, 13917, 
	    22148, 12767, 10978, 8652, 12123, 2801, 7357, 14269, 9455, 15919, 857, 24996, 
	    21670, 15852, 20621, 4858, 3253, 21335, 24858, 7316, 23145, 14378, 291, 20356, 
	    16383, 8422, 19021, 1332, 10772, 7437, 18103, 9035, 11962, 3770, 8981, 11434, 
	    9140, 10146, 11187, 21316, 16007, 11808, 9725, 1224, 4478, 3947, 23340, 10333, 
	    11954, 9263, 13982, 22041, 6862, 15654, 1722, 2565, 482, 24661, 16233, 9451, 
	    18030, 23928, 5582, 18871, 19683, 4370, 15291, 11763, 11201, 21915, 3495, 22219, 
	    1378, 16939, 17324, 11818, 12468, 1546, 13409, 11290, 19014, 9694, 13400, 7980, 
	    15911, 8735, 12715, 11263, 18628, 845, 23052, 6524, 17683, 14080, 8996, 1098, 
	    5586, 14167, 5670, 7670, 19745, 9524, 9143, 16078, 9722, 21100, 15762, 305, 
	    21072, 15316, 12689, 6084, 13197, 5317, 24963, 4289, 4621, 3071, 17307, 22618, 
	    7623, 2923, 9671, 19466, 16975, 19663, 24609, 2651, 2415, 9115, 7735, 12112, 29, 
	    3082, 3922, 21036, 19962, 4600, 18916, 368, 21484, 5962, 23609, 23323, 24482, 
	    8557, 19456, 14840, 7423, 10365, 21658, 2794, 16539, 14970, 19442, 16224, 8582, 
	    20022, 15126, 23253, 5806, 1539, 5271, 22883, 14024, 10255, 19660, 4837, 1054, 
	    22103, 676, 24053, 19636, 8275, 12261, 11886, 12434, 14401, 23708, 19087, 21373, 
	    21354, 6932, 8817, 6128, 13079, 16321, 9050, 14357, 9975, 15953, 21146, 8232, 
	    11554, 8448, 16015, 22390, 18455, 2055, 7048, 23546, 24507, 10965, 3672, 21956, 
	    2878, 11873, 22693, 15004, 21451, 20530, 16947, 24013, 2372, 24043, 23888, 
	    16123, 22573, 6239, 11558, 1992, 16433, 8644, 1940, 21688, 6660, 843, 13253, 
	    1763, 20005, 1218, 10799, 1717, 1451, 15297, 3597, 9321, 18220, 7467, 6464, 513, 
	    15064, 16459, 15864, 6633, 23777, 13181, 9710, 9448, 4815, 11313, 5720, 19301, 
	    18204, 20602, 21967, 17583, 7192, 9442, 21378, 12153, 22822, 15009, 18733, 6826, 
	    14962, 11998, 18419, 21173, 23927, 21037, 7490, 7438, 22888, 19870, 10771, 
	    22166, 22252, 8243, 16865, 19919, 8747, 3461, 6130, 11744, 1771, 20461, 12770, 
	    5981, 16440, 6646, 2455, 11162, 11050, 13264, 7515, 6702, 23917, 1139, 17823, 
	    19134, 7374, 14674, 16677, 1895, 8697, 5376, 16817, 10001, 20213, 17871, 7410, 
	    21652, 13962, 926, 3658, 7417, 1616, 13929, 1094, 14539, 7062, 23065, 5146, 
	    1060, 1544, 2578, 18462, 7519, 4367, 24034, 20783, 14482, 5381, 13762, 21259, 
	    6560, 21004, 8880, 19473, 4982, 2086, 11492, 22, 7907, 24788, 18776, 24275, 
	    7695, 12066, 21663, 7914, 9478, 18018, 19094, 13389, 981, 18333, 22269, 13955, 
	    16646, 23875, 1344, 13914, 17782, 13387, 5401, 4058, 19691, 1870, 18151, 15404, 
	    22050, 19954, 6280, 24582, 4248, 12057, 5771, 14440, 24456, 17796, 11338, 7826, 
	    18480, 8672, 23660, 7454, 9250, 10525, 7277, 1041, 53, 5991, 14975, 19622, 
	    23813, 24274, 2967, 20182, 18542, 22403, 21095, 14455, 1560, 13108, 8008, 14501, 
	    3963, 24453, 8605, 8415, 12235, 16164, 8655, 21626, 5156, 15382, 3044, 17814, 
	    1484, 13972, 19327, 7108, 11086, 21916, 1412, 15455, 21682, 22043, 11540, 5591, 
	    7608, 1971, 5305, 3611, 18080, 5091, 22907, 8567, 14630, 12450, 16310, 5337, 
	    17162, 6565, 4166, 21222, 4376, 6166, 22766, 9895, 12417, 2377, 6081, 20489, 
	    10390, 17415, 8916, 7990, 10876, 20172, 23348, 5230, 22692, 17231, 4876, 4049, 
	    14244, 5445, 11174, 21511, 6225, 16626, 19317, 4808, 881, 20776, 3608, 6736, 
	    3421, 7683, 19381, 7955, 7001, 24489, 9735, 21123, 17112, 7762, 9289, 18408, 
	    15655, 17923, 16564, 20447, 13265, 19178, 3157, 21438, 7260, 13780, 23768, 8535, 
	    7425, 13325, 10344, 678, 6332, 3946, 11774, 11061, 11984, 10172, 17560, 829, 
	    23555, 12411, 24007, 1764, 9900, 24613, 18096, 6670, 14685, 7985, 21931, 14109, 
	    6399, 17195, 4756, 17278, 5130, 3873, 14522, 19717, 3788, 6034, 7440, 13036, 
	    7655, 748, 14392, 23495, 4018, 21830, 20673, 15956, 9311, 9627, 10613, 8821, 
	    9465, 15695, 4482, 3709, 4328, 3581, 14521, 15033, 2200, 21008, 24146, 20836, 
	    7064, 5260, 9847, 24561, 3091, 14445, 17910, 24197, 21226, 16907, 22510, 14197, 
	    9183, 11119, 6769, 19486, 11653, 16192, 18525, 18194, 5914, 2441, 2963, 7061, 
	    2965, 20799, 10487, 24055, 4471, 17847, 8250, 12656, 20438, 18886, 20030, 16721, 
	    6555, 5422, 21001, 23584, 17778, 13630, 13335, 20137, 8478, 7820, 23533, 23193, 
	    22163, 12915, 22902, 9839, 1910, 15540, 7432, 13220, 20548, 8130, 10053, 4496, 
	    3738, 6001, 856, 20332, 18969, 2142, 1938, 196, 13493, 1840, 19286, 7022, 20793, 
	    9729, 764, 6538, 24725, 3245, 684, 3713, 22145, 129, 20435, 16556, 23025, 16774, 
	    22246, 13283, 19947, 19989, 24128, 12567, 12747, 6725, 10356, 10872, 1836, 8799, 
	    4816, 12458, 6695, 20595, 3979, 15487, 5326, 14098, 12917, 10307, 839, 6111, 
	    14655, 14324, 13051, 61, 3734, 14819, 10425, 10379, 8572, 21117, 3050, 16492, 
	    16253, 4734, 6517, 15421, 6211, 191, 18781, 8485, 8371, 24408, 1504, 704, 11615, 
	    21352, 13313, 8350, 17610, 9780, 22610, 5085, 17553, 11357, 12294, 18981, 17115, 
	    9286, 640, 9126, 23177, 982, 13439, 1525, 22824, 20710, 19358, 23882, 15606, 
	    5189, 21245, 9343, 21841, 19520, 15132, 13122, 18299, 20728, 18887, 7888, 11531, 
	    15722, 10148, 12443, 6933, 24232, 23290, 3966, 7172, 5583, 3514, 9952, 15507, 
	    3507, 17689, 2459, 18512, 7932, 20304, 8839, 22080, 18591, 12190, 23834, 6953, 
	    6807, 11814, 11858, 56, 22591, 8099, 23748, 1843, 18353, 9052, 5621, 15116, 
	    18058, 1237, 17968, 4193, 15391, 13872, 12991, 3061, 17248, 21564, 18092, 6116, 
	    23500, 24455, 23912, 14481, 24124, 19264, 24987, 2743, 8900, 695, 22343, 16128, 
	    7617, 3023, 11410, 6501, 4266, 5042, 7093, 12687, 4199, 9901, 13300, 17738, 
	    23550, 1936, 18793, 17144, 7111, 9609, 3520, 22830, 12597, 5447, 1117, 16139, 
	    19163, 9910, 9243, 18559, 17935, 2412, 11885, 1473, 11943, 19974, 19022, 2177, 
	    17343, 8562, 13199, 10582, 22782, 619, 23785, 4828, 20275, 2440, 14628, 21497, 
	    20081, 2873, 14349, 17666, 6650, 22720, 2280, 20012, 14366, 24131, 3140, 2850, 
	    10663, 11753, 16144, 24738, 522, 22969, 14188, 2158, 17235, 18001, 2808, 3560, 
	    9303, 16632, 23051, 10270, 24922, 18721, 15306, 2828, 12422, 24841, 2587, 3802, 
	    11756, 17234, 18789, 1501, 13569, 7057, 10903, 4923, 10685, 8835, 2011, 14405, 
	    18468, 7644, 23377, 24549, 23070, 2793, 685, 22265, 8971, 22974, 15954, 21237, 
	    15164, 8078, 18530, 12472, 17994, 12525, 22675, 11048, 10439, 19517, 1270, 
	    23779, 15392, 18158, 24139, 11002, 16096, 11477, 2101, 20544, 3766, 17780, 
	    16878, 12130, 4105, 10776, 23915, 10415, 7765, 15714, 6281, 5937, 98, 11024, 
	    5325, 2756, 583, 10971, 14590, 8418, 8217, 20388, 284, 3698, 1712, 20181, 22365, 
	    21529, 4359, 4757, 18849, 1921, 20845, 19541, 1176, 2456, 8999, 15244, 12903, 
	    2715, 5897, 23284, 9531, 6095, 10901, 7591, 13566, 22399, 14692, 17907, 884, 
	    21913, 16290, 6476, 13318, 24676, 22278, 11992, 22123, 4253, 3917, 9508, 17458, 
	    187, 9853, 1389, 9300, 8387, 12397, 21715, 6299, 15348, 18236, 18216, 6243, 
	    6021, 7628, 5428, 9859, 20318, 8688, 349, 23952, 19580, 12137, 11912, 7708, 
	    20840, 8804, 20290, 2960, 21811, 3306, 19574, 24071, 7419, 14718, 20330, 22410, 
	    14981, 5261, 14105, 15863, 20614, 12230, 9936, 21060, 10059, 24884, 1340, 2080, 
	    6547, 4733, 16967, 3127, 13842, 18528, 19266, 11003, 19039, 24248, 15285, 17062, 
	    7939, 12469, 17142, 20118, 2051, 22422, 3517, 10809, 8352, 6426, 3536, 20709, 
	    17321, 22344, 14396, 17557, 12428, 7310, 4520, 13458, 2203, 7633, 14871, 23961, 
	    15995, 19400, 22825, 10228, 11015, 15561, 19632, 338, 6708, 11054, 14498, 13268, 
	    514, 15385, 20993, 14740, 2933, 14945, 18423, 14572, 19402, 3410, 3570, 13320, 
	    3704, 2505, 24463, 17411, 20823, 20444, 9476, 9877, 7897, 22702, 16971, 3336, 
	    9964, 21366, 66, 18251, 22264, 3040, 22474, 1862, 2486, 5745, 5464, 15879, 
	    19918, 2645, 19123, 13672, 2977, 19265, 6136, 18054, 8323, 17640, 9716, 790, 
	    7912, 7929, 2314, 16133, 176, 11563, 11079, 22287, 6194, 10170, 13092, 11250, 
	    19604, 12150, 19411, 4515, 4452, 19521, 1993, 21513, 16367, 9495, 9946, 19601, 
	    17555, 12131, 15240, 16671, 6369, 24740, 21789, 8244, 8129, 20762, 16697, 4800, 
	    11207, 14032, 20267, 13127, 2741, 23298, 2394, 5463, 15881, 8650, 19214, 5750, 
	    20391, 10375, 329, 14403, 24910, 100, 21260, 9594, 17159, 288, 12784, 7876, 
	    19335, 3022, 7815, 11827, 3541, 21627, 23048, 12898, 11728, 20478, 21568, 7010, 
	    12317, 16094, 6648, 3016, 12107, 10856, 5423, 8501, 1750, 6737, 6413, 19058, 
	    13585, 22828, 5644, 21781, 2007, 4672, 18112, 23474, 10638, 7373, 7292, 15159, 
	    1464, 4549, 10759, 15791, 18654, 12857, 23460, 6335, 20471, 4530, 7972, 8146, 
	    22689, 3019, 816, 14627, 23766, 14762, 16275, 5994, 19231, 8781, 8380, 24797, 
	    15310, 8039, 6067, 3595, 9331, 11312, 21760, 17597, 6020, 21399, 3363, 9935, 
	    18690, 10848, 20045, 4776, 17991, 11217, 9107, 11810, 1458, 7645, 10193, 13857, 
	    16393, 15430, 17388, 20743, 22312, 24695, 21372, 15102, 17971, 2907, 8691, 
	    13776, 6174, 15687, 15368, 10114, 3126, 14421, 76, 6531, 24254, 365, 20729, 
	    1097, 21141, 9122, 7934, 9381, 21901, 13821, 7601, 467, 21187, 20885, 11557, 
	    19784, 5920, 15978, 9121, 9689, 16213, 9957, 11769, 8336, 12091, 42, 11974, 
	    19084, 14153, 24035, 19867, 12030, 10485, 2217, 15861, 3812, 10310, 24227, 6664, 
	    602, 8316, 6845, 22723, 8120, 4222, 9925, 4325, 5550, 12570, 6460, 2466, 4037, 
	    4378, 5534, 16563, 12534, 20126, 7777, 740, 137, 15314, 18897, 18564, 19120, 
	    3020, 4697, 22581, 23239, 24883, 24245, 6051, 6380, 24373, 16456, 10034, 7301, 
	    5681, 130, 9876, 19875, 9902, 6338, 278, 10526, 6288, 5824, 15275, 6876, 19128, 
	    24618, 9325, 16408, 21129, 2271, 12678, 23880, 11625, 21532, 5758, 21891, 24176, 
	    4784, 4774, 4085, 18798, 16384, 20180, 2497, 17805, 12394, 5622, 22977, 17457, 
	    19016, 6439, 16687, 14495, 21747, 6673, 6037, 2274, 4343, 23435, 18254, 11275, 
	    10730, 5662, 259, 20909, 12374, 190, 87, 22743, 13988, 12640, 17111, 17806, 
	    17480, 18956, 18740, 13354, 6066, 2869, 24830, 10075, 9200, 19761, 11236, 19697, 
	    16623, 15085, 19040, 1511, 3298, 1703, 3668, 19011, 2192, 11659, 24268, 22893, 
	    24823, 9736, 24342, 11287, 15364, 7080, 16963, 15160, 22393, 16785, 7496, 13973, 
	    23978, 17035, 9256, 23781, 19932, 19355, 23279, 3988, 20372, 3222, 24348, 14427, 
	    14554, 18854, 4140, 12070, 19161, 24093, 22511, 13752, 23400, 23442, 9897, 5787, 
	    12211, 9216, 5079, 6023, 16102, 240, 5850, 18233, 21374, 598, 21942, 46, 6728, 
	    12038, 14489, 8496, 24511, 22086, 16578, 22398, 17940, 10104, 14238, 12453, 
	    17898, 13148, 10167, 1144, 1877, 22340, 22039, 6971, 21742, 20086, 7265, 997, 
	    6024, 23451, 3128, 21204, 8774, 4617, 9622, 9569, 4082, 15292, 2205, 16705, 252, 
	    11052, 12548, 11291, 6126, 22349, 10211, 9450, 17879, 2620, 15147, 15068, 18301, 
	    21005, 6719, 20495, 14965, 22125, 17163, 1802, 21527, 20550, 22333, 18272, 
	    18796, 5457, 13554, 19602, 9114, 7040, 12918, 9253, 11527, 3147, 5439, 5360, 
	    14712, 19323, 9826, 5256, 7494, 10106, 17, 24684, 14748, 1304, 19863, 2889, 
	    6786, 16923, 19396, 10748, 7241, 5065, 15935, 9820, 24277, 7953, 16375, 12135, 
	    24444, 21883, 18475, 12882, 1140, 22924, 12462, 24665, 23023, 15026, 22873, 
	    5171, 12665, 1621, 10195, 18910, 13882, 6123, 12448, 9921, 18371, 23764, 20596, 
	    19079, 12396, 4432, 11617, 20698, 19158, 12404, 2603, 10209, 16249, 5813, 1888, 
	    15028, 14095, 23851, 23102, 20938, 21463, 2385, 4610, 5056, 18864, 5013, 17208, 
	    11596, 13446, 20561, 18387, 12129, 17057, 23299, 14473, 23720, 8126, 2130, 
	    14013, 19677, 8410, 4856, 22296, 185, 18649, 18048, 17856, 19367, 10137, 19192, 
	    10225, 20142, 19019, 2040, 2140, 1582, 9592, 13342, 4916, 12010, 2553, 6790, 
	    10942, 7037, 4316, 18727, 6920, 5752, 24892, 11584, 4657, 3552, 16980, 4372, 
	    10032, 9860, 8924, 24782, 135, 19657, 8454, 4263, 3787, 15177, 4071, 19687, 
	    12283, 17190, 9427, 9059, 7744, 20149, 15143, 1288, 10683, 3392, 2896, 12765, 
	    17310, 4572, 10666, 3559, 8255, 10887, 11074, 23948, 9055, 2984, 24783, 21617, 
	    17161, 15495, 13286, 17724, 24888, 17003, 5024, 9831, 3059, 11084, 11831, 17238, 
	    761, 6342, 21139, 5295, 15256, 9064, 594, 21731, 6562, 14151, 9846, 1424, 13299, 
	    3449, 2123, 9334, 16347, 7931, 7660, 19427, 20218, 812, 7562, 3233, 2834, 6599, 
	    13822, 4525, 54, 3228, 4271, 21828, 15295, 1171, 16106, 3721, 15501, 108, 17106, 
	    13693, 3142, 18599, 15582, 125, 19111, 15678, 1460, 1901, 21307, 4796, 14974, 
	    14668, 4533, 23891, 3362, 5289, 24779, 10070, 5712, 1267, 17711, 20488, 16141, 
	    24476, 23201, 3356, 20079, 23242, 3136, 21288, 7320, 13801, 13392, 14265, 14415, 
	    19569, 15021, 19371, 4006, 1069, 9885, 16716, 4412, 23869, 20629, 22651, 24167, 
	    14125, 10607, 9005, 8899, 21616, 7099, 11283, 17580, 10727, 23922, 19815, 14209, 
	    5409, 2964, 13974, 7413, 18739, 15059, 12222, 6120, 20672, 11793, 7087, 11869, 
	    7298, 5919, 24362, 12523, 5740, 19403, 11751, 15199, 3905, 18256, 9345, 23966, 
	    18742, 12663, 11632, 12259, 16511, 1752, 15842, 10470, 9917, 18382, 11365, 
	    10556, 11120, 11633, 24040, 3355, 18430, 1128, 950, 6080, 11155, 11436, 5884, 
	    6527, 10777, 11502, 22933, 1481, 7998, 471, 17153, 21413, 19460, 1000, 24078, 
	    5201, 9760, 24994, 11833, 2439, 22284, 20969, 10176, 10498, 11438, 21421, 2754, 
	    1517, 24950, 20780, 5757, 19649, 13742, 1147, 34, 14816, 4443, 13361, 16790, 
	    3349, 21698, 4421, 15058, 17550, 1442, 22332, 18772, 9690, 3213, 19160, 5782, 
	    23540, 13072, 14192, 17122, 1692, 12272, 9291, 20328, 23414, 11641, 2723, 16444, 
	    9849, 9659, 8724, 24363, 7297, 22407, 5094, 12578, 16236, 18852, 3489, 11621, 
	    8155, 2824, 22006, 16575, 18097, 358, 19113, 21207, 3027, 15631, 22175, 21765, 
	    16010, 2857, 5537, 15253, 16794, 21844, 7262, 2693, 12638, 5903, 22487, 6216, 
	    19647, 22727, 16261, 3445, 7763, 21552, 7919, 21479, 20283, 20343, 18587, 8163, 
	    22654, 15119, 22118, 19491, 9676, 31, 22609, 15169, 22188, 6391, 8121, 23080, 
	    15247, 9888, 14029, 1981, 19552, 23984, 18898, 15932, 3506, 8331, 11664, 18714, 
	    3351, 4100, 16065, 13865, 12541, 23588, 9817, 5857, 21130, 12224, 7911, 11694, 
	    15748, 19245, 20061, 22999, 3177, 16513, 18636, 1799, 4246, 17603, 24634, 18451, 
	    1193, 14897, 13324, 14622, 6816, 9236, 24295, 3730, 7201, 20705, 24435, 23544, 
	    1456, 12053, 3468, 12800, 15497, 2139, 10806, 1479, 4081, 16650, 15734, 6585, 
	    21912, 45, 5340, 19057, 21623, 4646, 362, 8169, 9822, 11781, 406, 7153, 13346, 
	    5490, 17980, 10774, 10438, 21816, 22074, 24567, 21856, 6793, 24422, 10740, 
	    16840, 20337, 1016, 14199, 19314, 10737, 14164, 449, 18337, 13245, 7526, 7513, 
	    515, 17251, 5895, 18009, 11958, 13166, 18784, 15420, 18375, 18460, 5046, 15223, 
	    14400, 6394, 9613, 5612, 11472, 8340, 7470, 16357, 20805, 490, 10931, 19012, 
	    966, 20358, 17408, 10734, 6525, 15219, 16542, 22706, 11848, 11606, 12690, 5135, 
	    10677, 23269, 4010, 11645, 19559, 11345, 2784, 23602, 8741, 19420, 17957, 138, 
	    17577, 4896, 2817, 6861, 21142, 1586, 20551, 11376, 21712, 19089, 20671, 3379, 
	    21332, 6595, 16957, 3724, 18082, 10423, 22850, 1354, 20722, 9223, 16277, 3526, 
	    2540, 10121, 19477, 19499, 21466, 21076, 7688, 15426, 5349, 3406, 20264, 8187, 
	    12728, 15328, 21978, 13804, 8555, 12399, 9324, 6404, 21542, 4160, 7731, 18019, 
	    9234, 11569, 3229, 4992, 13506, 17474, 20554, 19475, 983, 6219, 12596, 2916, 
	    15528, 11327, 5852, 10746, 8470, 5751, 6330, 17860, 21047, 19854, 2821, 348, 
	    15400, 22491, 8758, 742, 20594, 11286, 18708, 15819, 6888, 10154, 1791, 13934, 
	    3789, 4320, 16759, 13799, 19629, 14115, 1200, 8713, 12051, 17756, 7486, 9176, 
	    2721, 5421, 10249, 1770, 8249, 6200, 22760, 17348, 14790, 4966, 16941, 18131, 
	    1243, 21208, 22913, 16461, 24731, 7286, 13373, 2635, 19775, 3376, 12476, 23868, 
	    12966, 5705, 8370, 22884, 15449, 4477, 14092, 4004, 5638, 13182, 1488, 20169, 
	    16303, 457, 11846, 8911, 8041, 5357, 19770, 15985, 14510, 6282, 12816, 16431, 
	    11198, 7808, 24667, 5040, 12166, 11196, 23756, 2579, 1894, 6731, 22596, 9818, 
	    443, 10820, 733, 22856, 21244, 14128, 14066, 20908, 14556, 3203, 1776, 16116, 
	    24211, 4505, 16997, 7581, 2556, 20387, 9772, 18983, 9196, 14478, 22297, 18010, 
	    23496, 13416, 9819, 22623, 22097, 5034, 21202, 7787, 13578, 9488, 21482, 4472, 
	    6682, 6521, 6228, 21094, 20338, 8847, 2941, 8193, 3441, 22036, 8489, 5154, 
	    19912, 5632, 16481, 13073, 9158, 50, 17736, 12375, 22346, 7607, 225, 21721, 
	    23582, 14059, 4864, 12717, 132, 11534, 5942, 15806, 10637, 3488, 5827, 5083, 
	    23167, 832, 24555, 17446, 1495, 18547, 18197, 18319, 12769, 8145, 20232, 10207, 
	    22985, 16553, 12173, 13626, 10160, 21794, 14103, 22476, 4075, 86, 3442, 7039, 
	    17781, 23350, 3001, 16179, 17002, 18865, 9102, 4171, 3774, 9749, 20115, 17551, 
	    16174, 20984, 9803, 18838, 8063, 1489, 9708, 20627, 5608, 10215, 2465, 10235, 
	    2361, 11573, 7579, 21709, 15991, 4668, 6437, 11710, 13562, 19318, 11014, 12342, 
	    6328, 20872, 24926, 10944, 14843, 4302, 9106, 23029, 16483, 8770, 11515, 23902, 
	    22368, 6598, 24064, 20487, 12636, 6220, 3191, 20277, 9540, 9091, 1570, 13689, 
	    13128, 10159, 19861, 17312, 8443, 14409, 9572, 2214, 6512, 23295, 2761, 20155, 
	    13584, 22268, 15151, 14493, 10203, 22324, 15008, 10531, 14551, 19872, 6032, 
	    14968, 23195, 15380, 22570, 21941, 22621, 3512, 15727, 17615, 14155, 2208, 
	    21397, 15740, 14159, 15751, 5518, 3188, 21291, 14513, 10038, 3396, 1013, 5242, 
	    12085, 15517, 21415, 6361, 20930, 8974, 10152, 17113, 510, 9539, 12320, 14789, 
	    1564, 22649, 17236, 17616, 22081, 3556, 3003, 21134, 2670, 8953, 347, 4832, 
	    21215, 16103, 3223, 20985, 431, 23170, 6756, 8886, 6027, 1040, 14362, 680, 8766, 
	    5773, 24030, 8653, 13445, 18188, 2949, 21847, 21586, 18307, 21724, 11237, 11830, 
	    7590, 3576, 23998, 9543, 24075, 180, 1251, 7329, 10854, 15459, 5296, 24530, 
	    14483, 20675, 6609, 19728, 4853, 10477, 1962, 5113, 2195, 15773, 8252, 6541, 
	    2745, 12563, 10383, 11983, 17765, 11374, 8225, 199, 21325, 24607, 21957, 3728, 
	    14786, 10701, 9861, 12009, 24767, 13267, 4917, 7023, 1161, 23811, 17629, 15545, 
	    19626, 8715, 18652, 15980, 19713, 24653, 10502, 5095, 16282, 6100, 15473, 14708, 
	    7618, 18719, 24026, 9767, 4692, 6246, 14625, 21051, 10698, 6823, 22763, 11857, 
	    16618, 6317, 3329, 6260, 9528, 16195, 13431, 18076, 6103, 18208, 2707, 4149, 
	    21502, 7984, 15411, 24204, 71, 2059, 18243, 23027, 20399, 16569, 14190, 22521, 
	    15601, 10313, 2351, 10667, 16946, 1890, 6701, 6800, 23844, 15620, 11332, 1498, 
	    13893, 24925, 22388, 334, 21561, 14371, 14810, 16666, 21069, 2411, 19751, 22198, 
	    15494, 22353, 20806, 24207, 2549, 2036, 3288, 15800, 16239, 19833, 4571, 4739, 
	    16265, 5634, 5477, 13989, 8195, 20848, 481, 23328, 5531, 1684, 8379, 8526, 
	    11966, 17744, 3846, 19617, 23108, 6241, 16762, 1891, 20034, 9351, 5361, 7370, 
	    7946, 2287, 2667, 20053, 8459, 2781, 14620, 8098, 20726, 22373, 22673, 5489, 
	    16756, 23835, 17963, 17272, 1677, 2981, 5674, 6242, 6623, 6592, 3178, 15017, 
	    7540, 7580, 9705, 738, 9507, 21796, 6142, 12045, 10376, 13515, 15580, 5188, 
	    24499, 19518, 19220, 24946, 10028, 2699, 19984, 5314, 8867, 858, 17436, 8436, 
	    14947, 3671, 17473, 3313, 19899, 22611, 2926, 13871, 16932, 13310, 22153, 7334, 
	    5275, 2718, 20215, 11726, 2050, 10568, 10860, 12124, 22199, 15597, 16182, 5892, 
	    22602, 14449, 3777, 11799, 23216, 11812, 24861, 21934, 23726, 24249, 1173, 373, 
	    8377, 21780, 3978, 5654, 4890, 2726, 12631, 4136, 2301, 19298, 11775, 16887, 
	    6872, 2387, 3258, 2935, 18291, 20383, 9669, 19450, 4492, 18971, 19939, 3701, 
	    21782, 48, 10165, 3155, 5913, 4270, 19361, 16244, 23617, 10285, 24174, 12367, 
	    4491, 14763, 322, 9088, 23156, 15131, 15356, 21476, 1130, 376, 3399, 2650, 9984, 
	    12936, 18999, 5749, 2332, 12838, 16121, 1552, 16162, 19315, 3463, 3367, 8654, 
	    23199, 11566, 12496, 21560, 23085, 20880, 7102, 1619, 24399, 2420, 10940, 6636, 
	    11657, 7881, 9634, 22220, 24483, 4794, 5021, 16148, 20515, 3343, 22544, 22280, 
	    990, 18065, 6751, 18040, 18756, 15228, 6321, 1640, 19006, 7409, 19223, 10946, 
	    3848, 888, 5246, 7101, 21649, 4457, 11658, 14936, 14326, 5321, 21947, 22849, 
	    15572, 19490, 20357, 11771, 24012, 5998, 8937, 8226, 19549, 8594, 4351, 15670, 
	    13778, 1668, 12652, 2518, 20715, 7676, 22538, 3208, 21750, 10953, 6041, 20196, 
	    21598, 9172, 731, 22085, 7154, 7341, 15390, 11385, 6973, 775, 7362, 8504, 7257, 
	    141, 22780, 15843, 24044, 17934, 1620, 15792, 9536, 8229, 23549, 12698, 1044, 
	    4445, 11820, 15993, 6799, 12571, 1610, 11065, 1824, 22588, 20235, 4949, 15616, 
	    13698, 20824, 5746, 1059, 20396, 28, 22894, 16407, 11590, 20147, 6240, 6096, 
	    15810, 15854, 20389, 10322, 21845, 18317, 12575, 6766, 9977, 4300, 502, 14198, 
	    21955, 24302, 22541, 17692, 22940, 15386, 20075, 19923, 24395, 4431, 8171, 
	    12108, 16816, 5442, 425, 16831, 23321, 18773, 16317, 13686, 22738, 20237, 20634, 
	    20980, 22838, 10923, 5110, 3010, 17462, 2671, 8561, 10747, 11326, 20428, 16308, 
	    6236, 8796, 22869, 9197, 20761, 10436, 18288, 9504, 421, 1003, 4336, 22456, 
	    24149, 20103, 8975, 23523, 7097, 8532, 4040, 8935, 24394, 4601, 554, 6965, 
	    22203, 21023, 21768, 11146, 24891, 11737, 3186, 23397, 2833, 15188, 18604, 
	    19127, 3775, 7805, 10388, 1785, 4844, 10346, 13605, 438, 17041, 24058, 8023, 
	    18354, 23931, 18885, 7668, 24747, 19792, 9388, 24223, 5107, 14468, 19892, 20866, 
	    1042, 15899, 13573, 15262, 19681, 13213, 23292, 18848, 8646, 24784, 21732, 
	    23382, 16232, 8788, 8529, 6147, 14586, 24805, 17289, 19038, 14237, 10017, 10191, 
	    519, 20367, 7962, 16475, 15962, 17928, 22117, 16218, 19027, 17091, 12350, 19350, 
	    21647, 9166, 24796, 17793, 8871, 2929, 16191, 17674, 12909, 7978, 15117, 5309, 
	    2182, 24490, 19986, 13174, 24873, 7572, 22342, 5731, 2546, 181, 8373, 23053, 
	    6859, 16522, 22355, 949, 9755, 7495, 2563, 6946, 2079, 16405, 12451, 4908, 
	    15448, 21459, 22357, 13978, 7228, 15078, 19005, 21674, 17893, 21864, 4201, 
	    21819, 15901, 4350, 20281, 2544, 6916, 10733, 5411, 16586, 22341, 7205, 22633, 
	    12620, 13784, 21634, 8665, 8772, 8399, 7252, 5931, 5331, 15293, 16603, 15918, 
	    4474, 17995, 13112, 419, 19882, 5254, 9037, 9255, 6977, 2567, 11697, 21184, 
	    1814, 5497, 19772, 23398, 5299, 14260, 19050, 21055, 14549, 7489, 4210, 4717, 
	    1152, 14315, 18173, 18101, 22569, 8104, 12749, 18070, 14891, 18631, 16111, 8613, 
	    3238, 9605, 800, 2215, 9266, 19597, 19853, 9678, 12627, 16637, 788, 16812, 411, 
	    4919, 4518, 9597, 22214, 1150, 22431, 11132, 17543, 22480, 20718, 1670, 3383, 
	    18269, 13140, 24460, 9996, 18648, 13928, 22997, 4990, 2969, 24719, 23246, 23933, 
	    1257, 15699, 12851, 9673, 14953, 5995, 10650, 23226, 1744, 11709, 9305, 9712, 
	    5680, 20217, 16388, 20604, 21419, 17735, 12285, 935, 11602, 4130, 5985, 15032, 
	    15234, 17874, 22579, 2554, 6132, 24126, 20523, 19092, 13395, 9670, 5342, 23610, 
	    15186, 20365, 3429, 21546, 14902, 13440, 24696, 12146, 14318, 1666, 9150, 17670, 
	    3602, 24417, 1381, 4117, 11278, 16188, 24641, 540, 5889, 10337, 17005, 22558, 
	    21851, 6909, 21296, 19666, 17365, 489, 11285, 11739, 8321, 18095, 11346, 7967, 
	    7443, 8524, 5270, 20190, 22255, 5392, 6656, 13005, 16409, 3771, 17802, 8488, 
	    8968, 21565, 9191, 11190, 19528, 485, 4364, 12032, 11864, 16782, 13171, 5898, 
	    11087, 24011, 21608, 9978, 11719, 21147, 1672, 23115, 17028, 23208, 18684, 
	    17333, 5371, 6743, 3170, 16122, 5566, 21857, 21879, 9924, 20746, 12480, 20581, 
	    15083, 23493, 16798, 8849, 10491, 13514, 21064, 18472, 3227, 10099, 9467, 3411, 
	    24224, 19907, 12321, 20574, 13358, 214, 10520, 15682, 12633, 13121, 13816, 
	    17628, 23877, 19530, 10911, 21935, 6210, 8412, 15020, 12818, 14652, 19612, 
	    24683, 1638, 7892, 22411, 10128
	};
	
	/**
	 * Do we want:
	 * req==0: Random selection
	 * req==1: Sequential selection
	 */
	
	uint64_t n=0;
	if(pkt->req == 0){
		do{
			n = imgsConnRandom(connCtx, pkt->first, pkt->last);
			n = SPLIT[n];
		}while(n == 24876 || /* Non-dog, non-cat */
		       n == 10360 || /* Dupe */
		       n == 15582 || /* Dupe */
		       n == 22901 || /* Dupe */
		       n == 23297 || /* Dupe */
		       n ==  6204 || /* Dupe */
		       n ==  2339 ); /* Dupe */
	}else if(pkt->req == 1){
		n = pkt->first + i;
		n = SPLIT[n];
	}
	
	return n;
}

/**
 * Mark connection as requesting exit.
 */

void imgsConnWantExit(CONN_CTX* connCtx){
	connCtx->exit = 1;
}

/**
 * Was exit requested?
 */

int  imgsConnIsExitWanted(CONN_CTX* connCtx){
	return connCtx->exit;
}

/**
 * Teardown connection.
 */

void imgsConnTeardown(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	unsigned char* addr = (unsigned char*)&connCtx->remoteAddr.sin_addr.s_addr;
	unsigned       port = htons(connCtx->remoteAddr.sin_port);
	printf("Closing connection from %u.%u.%u.%u:%u\n",
	       addr[0], addr[1], addr[2], addr[3], port);
	cudaFree(connCtx->devBounceBuf);
	close(connCtx->sockFd);
}

/**
 * Handle the connection's lifetime.
 */

void imgsConnHandle(GLOBAL_DATA* gData, CONN_CTX* connCtx){
	if(imgsConnSetup(gData, connCtx) < 0){
		printf("Could not set up resources for new connection!\n");
		imgsConnWantExit(connCtx);
	}
	
	while(!imgsConnIsExitWanted(connCtx)){
		imgsConnHandlePacket(gData, connCtx);
	}
	
	imgsConnTeardown(gData, connCtx);
}

/**
 * Check and parse argument sanity
 */

int  imgsParseAndCheckArgs(GLOBAL_DATA* gData){
	if(gData->argc != 2 && gData->argc != 3){
		printf("Usage: imgserver <path/to/catsanddogs.hdf5> {TCPPort#}\n");
		return -1;
	}
	
	/* File accessible? */
	struct stat fileStat;
	if(stat(gData->argv[1], &fileStat) < 0){
		printf("'%s' cannot be accessed!\n", gData->argv[1]);
		return -1;
	}
	
	/* File is a regular file? */
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

int  imgsGlobalSetup(GLOBAL_DATA* gData){
	/* Register callback */
	signal(SIGHUP,  sigHandler);
	signal(SIGINT,  sigHandler);
	signal(SIGQUIT, sigHandler);
	signal(SIGTERM, sigHandler);
	signal(SIGABRT, sigHandler);
	signal(SIGSEGV, sigHandler);
	atexit(atexitHandler);
	
	/* Argument sanity check */
	if(imgsParseAndCheckArgs(gData) < 0){
		printf("Arguments are insane!\n");
		return -1;
	}
	
	/* Create server socket */
	if(imgsNetworkSetup(gData) < 0){
		printf("Failure in network setup!\n");
		return -1;
	}
	
	/* Load data */
	if(imgsDataSetup(gData) < 0){
		printf("Failure in data load!\n");
		return -1;
	}
	
	/* Setup CUDA. Must happen after HDF5 data load. */
	if(imgsCUDASetup(gData) < 0){
		printf("Failure in CUDA setup!\n");
		return -1;
	}
	
	return 0;
}

/**
 * Global teardown.
 */

void imgsGlobalTeardown(GLOBAL_DATA* gData){
	if(gData->exiting == 0){
		gData->exiting = 1;
		imgsCUDATeardown(gData);
		imgsDataTeardown(gData);
		imgsNetworkTeardown(gData);
		gData->exiting = 2;
	}
}

/**
 * SIGINT interrupt handler.
 */

void sigHandler(int sig){
	imgsGlobalTeardown(&gData);
}

/**
 * atexit() handler.
 */

void atexitHandler(void){
	imgsGlobalTeardown(&gData);
}

/**
 * Event loop.
 * 
 * Accept client, handle requests.
 */

int imgsEventLoop(GLOBAL_DATA* gData){
	CONN_CTX* connCtx;
	
	printf("Accepting clients on port %d\n", gData->localPort);
	while((connCtx = imgsConnAccept(gData))){
		imgsConnHandle(gData, connCtx);
		free(connCtx);
	}
	
	return 0;
}


/**
 * Main
 */

int main(int argc, char* argv[]){
	int             ret;
	gData.argc    = argc;
	gData.argv    = (const char**)argv;
	gData.exiting = 0;
	
	/* Global setup */
	ret = imgsGlobalSetup(&gData);
	if(ret < 0){
		printf("Failure in setup!\n");
		return ret;
	}
	
	/* Run server till it dies. */
	ret = imgsEventLoop(&gData);
	
	/* Tear down. */
	imgsGlobalTeardown(&gData);
	
	/* Return */
	return ret;
}

