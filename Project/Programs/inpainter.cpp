/* Includes */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>



/* Namespaces in use */
using namespace std;
using namespace cv;



/**
 * Main
 */

int main(int argc, char* argv[]){
	/* Check argument */
	if(argc != 3){
		printf("Please provide path to train directory as the first argument,\n"
		       "and a target size as the second argument.\n");
		return 1;
	}
	
	/* Target size is 256x256. */
	const size_t T = strtoul(argv[2], 0, 0);
	const int    dilateRadius  = 3;
	const int    inpaintRadius = 10;
	
	/* Open files. */
	FILE* trainx            = fopen("trainx.bin",   "wb+");//shape=(25000,256,256,3), dtype="uint8"
	FILE* trainy            = fopen("trainy.bin",   "wb+");//shape=(25000,2),         dtype="float32"
	
	
	/**
	 * Expand those files to the correct tensor size.
	 */
	
	ftruncate(fileno(trainx),   25000*T*T*3*sizeof(uint8_t));
	ftruncate(fileno(trainy),   25000*2*    sizeof(float));
	
	
	/* Progress tracker. */
	int imgsdone = 0;
	
	/* Inpaint those images. */
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(size_t i=0;i<25000;i++){
		/* Get image path */
		int         catOrDog    = i<12500;
		const char* catOrDogStr = catOrDog ? "cat" : "dog";
		int         imgNum      = (catOrDog ? i : i-12500);
		char        imgpath[1024];
		sprintf(imgpath, "%s/%s.%d.jpg", argv[1], catOrDogStr, imgNum);
		
		
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
		
		
		if(0){
			printf("%dx%d\n", img.rows, img.cols);
			fflush(stdout);
			imshow("Test", img);
			waitKey();
		}
		
		/* RESULTS WRITEBACK */
		/* Seek and write to all files. */
#pragma omp critical
		{
			/* Sanity checks. */
			if(img.rows != T || img.cols != T || img.step[0] != 3*T){
				printf("ERROR! Bad Size!\n");
				fflush(stdout);
				exit(1);
			}
			
			/* Output. */
			/* x */
			fseek (trainx,    i*T*T*3*sizeof(uint8_t), SEEK_SET);
			fwrite(img.data, 1, T*T*3*sizeof(uint8_t), trainx);
			
			/* y */
			float y[2] = {catOrDog ? 1.0 : 0.0,
			              catOrDog ? 0.0 : 1.0};
			fseek(trainy, i*2*sizeof(float), SEEK_SET);
			fwrite(y, 1, 2*sizeof(float), trainy);
			
			/* Flush */
			fflush(trainx);
			fflush(trainy);
			
			/* Atomically increment counter of images done */
			imgsdone++;
			
			/* Print progress */
			printf("\rImage: [%5d/25000]", imgsdone);
			fflush(stdout);
		}
	}
	
	/* Newline. */
	printf("\n");
	
	/* Close files and exit. */
	fclose(trainx);
	fclose(trainy);
	
	return 0;
}

