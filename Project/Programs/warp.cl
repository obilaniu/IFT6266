/**
 * Image Warping Kernel.
 */

kernel void warpBlock4x4(__global   float*           dst,
                         read_only  image2d_array_t  src,
                         __global   const float*     Harray,
                         unsigned int                IPERTHRD,
                         unsigned int                JBLKPERTHRD,
                         unsigned int                KBLKPERTHRD,
                         unsigned int                ROWPITCH,
                         unsigned int                SLICEPITCH){
	/* Get IDs. */
	size_t gid0  = get_global_id(0);
	size_t gid1  = get_global_id(1);
	size_t gid2  = get_global_id(2);
	
	
	/* Block coordinates to be processed. */
	const unsigned int BLKH       = 4;                       /* Height of block to be processed. Always 4. */
	const unsigned int BLKW       = 4;                       /* Width  of block to be processed. Always 4. */
	const unsigned int iStart     = (gid2 + 0) * IPERTHRD;   /* Inclusive first image        to process. */
	const unsigned int iEnd       = (gid2 + 1) * IPERTHRD;   /* Exclusive last  image        to process. */
	const unsigned int jStart     = (gid1 + 0) * JBLKPERTHRD;/* Inclusive first row    block to process. */
	const unsigned int jEnd       = (gid1 + 1) * JBLKPERTHRD;/* Exclusive last  row    block to process. */
	const unsigned int kStart     = (gid0 + 0) * KBLKPERTHRD;/* Inclusive first column block to process. */
	const unsigned int kEnd       = (gid0 + 1) * KBLKPERTHRD;/* Exclusive last  column block to process. */
	const unsigned int rowPitch   = ROWPITCH;                /* Number of elements to go from one row to the next. */
	const unsigned int slicePitch = SLICEPITCH;              /* Number of elements to go from one image to the next. */
	
	/* Sampler */
	const sampler_t smpl = CLK_NORMALIZED_COORDS_FALSE |
	                       CLK_ADDRESS_CLAMP           |
	                       CLK_FILTER_LINEAR           ;
	
	
	/**
	 * Master Loop. At each innermost loop iteration, we process a 4x4 block.
	 */
	 
	__global float(*H)[3][3] = (__global float(*)[3][3])Harray;
	for(unsigned int i=iStart;i<iEnd;i++){        /* Loop over images. */
		/**
		 * Load homography for this image.
		 * 
		 * Every three planes share the same H.
		 */
		
		float H00 = H[i/3][0][0];
		float H01 = H[i/3][0][1];
		float H02 = H[i/3][0][2];
		float H10 = H[i/3][1][0];
		float H11 = H[i/3][1][1];
		float H12 = H[i/3][1][2];
		float H20 = H[i/3][2][0];
		float H21 = H[i/3][2][1];
		float H22 = H[i/3][2][2];
		
		for(unsigned int j=jStart;j<jEnd;j++){    /* Loop over row blocks. */
			for(unsigned int k=kStart;k<kEnd;k++){/* Loop over column blocks. */
				/* Compute block base coordinates */
				unsigned int yBase = j*BLKH;
				unsigned int xBase = k*BLKW;
				
				/* Compute homogeneous block coordinates */
				float16 y          = yBase + (float16)(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
				float16 x          = xBase + (float16)(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3);
				
				/**
				 * Warp the coordinates through the homography.
				 * 
				 *     [ xp ]   [ H00 H01 H02 ] [ x ]
				 *     [ yp ] = [ H10 H11 H12 ] [ y ]
				 *     [ zp ]   [ H20 H21 H22 ] [1.0]
				 */
				 
				float16 xp         = H00*x + H01*y + H02;
				float16 yp         = H10*x + H11*y + H12;
				float16 zp         = H20*x + H21*y + H22;
				xp /= zp;
				yp /= zp;
				
				/* Perform the Texture Fetches. */
				float16 v = (float16)(
				    read_imagef(src, smpl, (float4)(xp.s0,yp.s0,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s1,yp.s1,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s2,yp.s2,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s3,yp.s3,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s4,yp.s4,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s5,yp.s5,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s6,yp.s6,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s7,yp.s7,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s8,yp.s8,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.s9,yp.s9,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.sa,yp.sa,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.sb,yp.sb,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.sc,yp.sc,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.sd,yp.sd,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.se,yp.se,i,0)).x,
				    read_imagef(src, smpl, (float4)(xp.sf,yp.sf,i,0)).x
				);
				
				/* Write out the fetched values. */
				vstore4(v.s0123, 0, dst + i*slicePitch + (yBase+0)*rowPitch + xBase);
				vstore4(v.s4567, 0, dst + i*slicePitch + (yBase+1)*rowPitch + xBase);
				vstore4(v.s89ab, 0, dst + i*slicePitch + (yBase+2)*rowPitch + xBase);
				vstore4(v.scdef, 0, dst + i*slicePitch + (yBase+3)*rowPitch + xBase);
			}
		}
	}
}
