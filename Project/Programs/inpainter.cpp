/* Includes */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <hdf5.h>
#include <omp.h>


/* Defines */
#define NUM_IMAGES 25000



/* Namespaces in use */
using namespace std;
using namespace cv;


/* Functions */

/**
 * HDF5 file opening.
 * 
 * Opens the desired HDF5 file. Also sets alignment parameters.
 */

hid_t openHDF5File(const char* destFile){
	/* Want large datasets aligned on MB boundaries. */
	hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_alignment(plist, 16U<<10,  4U<<10);
	H5Pset_alignment(plist, 64U<<20, 16U<<20);
	
	/* Create file with given alignment configurations. */
	hid_t f     = H5Fcreate(destFile, H5F_ACC_EXCL, H5P_DEFAULT, plist);
	H5Pclose(plist);
	if(f<0){
		printf("ERROR! Failed to create HDF5 file %s!\n", destFile);
		return f;
	}
	
	/* Create group /data. */
	hid_t fdata = H5Gcreate2(f, "/data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(fdata<0){
		printf("ERROR! Failed to create group /data!\n");
		H5Fclose(f);
		return fdata;
	}
	
	/* Close handle to that group. */
	H5Gclose(fdata);
	H5Fflush(f, H5F_SCOPE_GLOBAL);
	
	/* Return file handle. */
	return f;
}


/**
 * Process one image size T, and insert it into the file f under
 *     /data/x_TxT
 */

int  processX(const char* sourceDir,
              int         dilateRadius,
              int         inpaintRadius,
              size_t      T,
              hid_t       f){
	/**
	 * Construct HDF5 dataset that is:
	 * 
	 * - Named /data/x_<Height>x<Width>
	 * - uint8
	 * - Dimension NUM_IMAGESx3x<Height>x<Width>
	 */
	
	char xdatasetPath[80];
	sprintf(xdatasetPath, "/data/x_%dx%d", T, T);
	hsize_t xdims[] = {NUM_IMAGES, 3, T, T};
	hid_t   xdspace = H5Screate_simple(4, xdims, NULL);
	hid_t   xdset   = H5Dcreate2(f, xdatasetPath, H5T_STD_U8LE, xdspace,
	                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Fflush(xdset, H5F_SCOPE_GLOBAL);
	if(xdset<0){
		H5Sclose(xdspace);
		printf("ERROR! Failed to create dataset %s!\n", xdatasetPath);
		return -1;
	}
	
	
	
	/* Progress tracker. */
	int imgsdone = 0;
	
	
	
	/**
	 * Inpaint those images in parallel.
	 */
	
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(size_t j=0;j<NUM_IMAGES;j++){
		/* Get image path */
		int         catOrDog    = j<(NUM_IMAGES/2);
		const char* catOrDogStr = catOrDog ? "cat" : "dog";
		int         imgNum      = (catOrDog ? j : j-(NUM_IMAGES/2));
		char        imgpath[1024];
		sprintf(imgpath, "%s/%s.%d.jpg", sourceDir, catOrDogStr, imgNum);
		
		
		/* Read images */
		Mat img = imread(imgpath);
		
		
		/**
		 * Scale Robustly.
		 * 
		 * We use OpenCV's pyrUp or pyrDown until we are within 2x of the target scale,
		 * then we resize linearly.
		 */
		
		do{
			int maxDim = img.cols > img.rows ? img.cols : img.rows;
			if(maxDim/2 >= T){
				/* We need to downsample (go down a pyramid level). */
				pyrDown(img, img);
			}else if(maxDim*2 <= T){
				/* We need to upsample (go up a pyramid level). */
				pyrUp(img, img);
			}else{
				/**
				 * We're close to the target size. We cubically resize to the target
				 * dimension, then break. At least one of img.cols or img.rows will
				 * equal T once we break out of the loop.
				 */
				
				double scaleFactor = (double)T/maxDim;
				size_t U;
				if(img.cols > img.rows){
					/**
					 * Image wider than it is tall. Final image will have 256
					 * cols and less than that rows.
					 */
					U = img.rows * scaleFactor;
					resize(img, img, Size(T,U), 0, 0, INTER_CUBIC);
				}else{
					U = img.cols * scaleFactor;
					resize(img, img, Size(U,T), 0, 0, INTER_CUBIC);
				}
				break;
			}
		}while(1);
		
		
		/**
		 * Inpaint the borders if needed, so that the image is TxT in size.
		 */
		
		if(img.rows == T && img.cols == T){
			/* Do nothing. */
		}else{
			Mat padded;
			Mat inpaintMask;
			
			if(img.rows == T){
				/**
				 * Image needs to be centered horizontally and the vertical bands
				 * left and right must be inpainted.
				 */
				
				/* 1. Compute padding. */
				int padL = (T-img.cols)/2;
				int padR = T-img.cols-padL;
				
				/**
				 * 2. Pad image with black to the final size,
				 *    centering properly.
				 */
				copyMakeBorder(img,                             /* Start with the image... */
							   padded,                          /* Put the result into the padded image... */
							   0, 0, padL, padR,                /* Padding top, bottom, left and right as required... */
							   BORDER_CONSTANT, 0);             /* With a padding of black. */
				
				/**
				 * 3. Make an inpainting mask of the same size as the padded image,
				 *    setting cells that must be untouched to black and the pixels that
				 *    must be predicted (padding) to white.
				 */
				copyMakeBorder(Mat::zeros(img.size(), CV_8UC1), /* Start with a mask of zeros... */
							   inpaintMask,                     /* Put the result into the inpaint mask... */
							   0, 0, padL, padR,                /* Padding top, bottom, left and right as required... */
							   BORDER_CONSTANT, 255);           /* With a padding of white. */
				
				/* 4. Inpaint. */
				dilate(inpaintMask, inpaintMask, Mat::ones(2*dilateRadius+1,2*dilateRadius+1,CV_8UC1));
				inpaint(padded, inpaintMask, img, inpaintRadius, INPAINT_TELEA);
			}else if(img.cols == T){
				/**
				 * Image needs to be centered vertically and the horizontal bands
				 * above and below must be inpainted.
				 */
				
				/* 1. Compute padding. */
				int padT = (T-img.rows)/2;
				int padB = T-img.rows-padT;
				
				/**
				 * 2. Pad image with black to the final size,
				 *    centering properly.
				 */
				copyMakeBorder(img,                             /* Start with the image... */
							   padded,                          /* Put the result into the padded image... */
							   padT, padB, 0, 0,                /* Padding top, bottom, left and right as required... */
							   BORDER_CONSTANT, 0);             /* With a padding of black. */
				
				/**
				 * 3. Make an inpainting mask of the same size as the padded image,
				 *    setting cells that must be untouched to black and the pixels that
				 *    must be predicted (padding) to white.
				 */
				copyMakeBorder(Mat::zeros(img.size(), CV_8UC1), /* Start with a mask of zeros... */
							   inpaintMask,                     /* Put the result into the inpaint mask... */
							   padT, padB, 0, 0,                /* Padding top, bottom, left and right as required... */
							   BORDER_CONSTANT, 255);           /* With a padding of white. */
				
				/* 4. Inpaint. */
				dilate(inpaintMask, inpaintMask, Mat::ones(2*dilateRadius+1,2*dilateRadius+1,CV_8UC1));
				inpaint(padded, inpaintMask, img, inpaintRadius, INPAINT_TELEA);
			}else{
				printf("ERROR! Inpainting should have at least one dimension at target!\n"
					   "Currently %dx%d !\n", img.rows, img.cols);
				fflush(stdout);
				exit(1);
			}
		}
		
		
		/**
		 * Split image into three planes for storage.
		 */
		
		vector<Mat> channs;
		split(img, channs);
		
#ifndef _OPENMP
		printf("%dx%d\n", img.rows, img.cols);
		fflush(stdout);
		imshow("Img", img);
		imshow("B", channs[0]);
		imshow("G", channs[1]);
		imshow("R", channs[2]);
		waitKey();
#endif
		
		
		/* RESULTS WRITEBACK */
		/* Seek and write to the dataset. */
#pragma omp critical
		{
			/* Sanity checks. */
			if(img.rows != T || img.cols != T ||
			   channs[0].step[0] != T         ||
			   channs[1].step[0] != T         ||
			   channs[2].step[0] != T         ){
				printf("ERROR! Bad Size!\n");
				fflush(stdout);
				exit(1);
			}
			
			/* Output. */
			hsize_t sliceDim[]    = {1,1,T,T};
			hsize_t sliceStart[]  = {0,0,0,0};
			hsize_t sliceStart0[] = {j,0,0,0};
			hsize_t sliceStart1[] = {j,1,0,0};
			hsize_t sliceStart2[] = {j,2,0,0};
			
			
			hid_t sliceDspace = H5Screate_simple(4, sliceDim, NULL);
			H5Sselect_hyperslab(sliceDspace, H5S_SELECT_SET,
			                    sliceStart, NULL, sliceDim, NULL);
			
			H5Sselect_hyperslab(xdspace, H5S_SELECT_SET,
			                    sliceStart0, NULL, sliceDim, NULL);
			H5Dwrite(xdset, H5T_NATIVE_UINT8, sliceDspace, xdspace,
			         H5P_DEFAULT, channs[0].data);
			H5Sselect_hyperslab(xdspace, H5S_SELECT_SET,
			                    sliceStart1, NULL, sliceDim, NULL);
			H5Dwrite(xdset, H5T_NATIVE_UINT8, sliceDspace, xdspace,
			         H5P_DEFAULT, channs[1].data);
			H5Sselect_hyperslab(xdspace, H5S_SELECT_SET,
			                    sliceStart2, NULL, sliceDim, NULL);
			H5Dwrite(xdset, H5T_NATIVE_UINT8, sliceDspace, xdspace,
			         H5P_DEFAULT, channs[2].data);
			H5Sclose(sliceDspace);
			
			
			/* Atomically increment counter of images done */
			imgsdone++;
			
			/* Print progress */
			printf("\rImage %dx%d: [%5d/%5d]", T, T, imgsdone, NUM_IMAGES);
			fflush(stdout);
		}
	}
	
	
	/* Newline. */
	printf("\n");
	printf("Created dataset %s @ 0x%016lx\n", xdatasetPath, H5Dget_offset(xdset));
	
	
	/* Close dataset */
	H5Fflush(xdset, H5F_SCOPE_GLOBAL);
	H5Sclose(xdspace);
	H5Dclose(xdset);
	
	/* Return successfully */
	return 0;
}

/**
 * Process the Y targets, and insert them into the file f under
 *     /data/y
 */

int  processY(hid_t f){
	/**
	 * Allocate /data/y
	 */
	
	hsize_t ydims[] = {NUM_IMAGES, 2};
	hid_t   ydspace = H5Screate_simple(2, ydims, NULL);
	hid_t   ydset   = H5Dcreate2(f, "/data/y", H5T_IEEE_F32LE, ydspace,
	                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Fflush(ydset, H5F_SCOPE_GLOBAL);
	if(ydset<0){
		H5Sclose(ydspace);
		printf("ERROR! Failed to create dataset /data/y!\n");
		return ydset;
	}
	
	
	/**
	 * Write out the data.
	 */
	
	for(int j=0;j<NUM_IMAGES;j++){
		/* Calculate distribution... */
		int   catOrDog = j<(NUM_IMAGES/2);
		float y[2] = {catOrDog ? 1.0 : 0.0,
		              catOrDog ? 0.0 : 1.0};
		
		/*  Write out at correct offset. */
		hsize_t ysliceDim[]    = {1,2};
		hsize_t ysliceStart[]  = {0,0};
		hsize_t ysliceStart0[] = {j,0};
		hid_t   sliceDspace    = H5Screate_simple(2, ysliceDim, NULL);
		H5Sselect_hyperslab(sliceDspace, H5S_SELECT_SET,
		                    ysliceStart, NULL, ysliceDim, NULL);
		
		
		H5Sselect_hyperslab(ydspace, H5S_SELECT_SET,
		                    ysliceStart0, NULL, ysliceDim, NULL);
		H5Dwrite(ydset, H5T_NATIVE_FLOAT, sliceDspace, ydspace,
		         H5P_DEFAULT, y);
		H5Sclose(sliceDspace);
	}
	
	/* Close */
	printf("Created dataset /data/y @ 0x%016lx\n", H5Dget_offset(ydset));
	H5Fflush(ydset, H5F_SCOPE_GLOBAL);
	H5Sclose(ydspace);
	H5Dclose(ydset);
}


/**
 * Main
 */

int main(int argc, char* argv[]){
	/* Check argument */
	if(argc != 3){
		printf("Please provide the path to the train directory as the first"
		       "argument,\nand a target file as the second argument.\n");
		return -1;
	}
	
	/* Inpainting parameters */
	const char*  sourceDir     = argv[1];
	const char*  destFile      = argv[2];
	const int    dilateRadius  = 3;
	const int    inpaintRadius = 10;
	const size_t NUM_T         = 3;
	const size_t Tarr[NUM_T]   = {64, 128, 256};
	//const size_t NUM_T         = 1;
	//const size_t Tarr[NUM_T]   = {64};
	
	
	/**
	 * HDF5 file opening.
	 */
	
	hid_t f = openHDF5File(destFile);
	if(f<0){goto quit;}
	
	
	/**
	 * For all desired image sizes:
	 */
	
	for(int i=0;i<NUM_T;i++){
		if(processX(sourceDir, dilateRadius, inpaintRadius, Tarr[i], f)<0){goto quit;}
	}
	
	/**
	 * For output targets:
	 */
	
	if(processY(f)<0){goto quit;}
	
	/* Close files and exit. */
	quit:
	H5Fclose(f);
	
	return 0;
}

