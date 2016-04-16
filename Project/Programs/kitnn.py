#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module Docstring
"""
KITNN: The KITten Neural Network, by Olexa Bilaniuk.

A corny name, as is customary for computer scientists, given to my CNN that
attempts to solve the problem of correctly classifying Cats & Dogs for the
Kaggle challenge. This work is done as part of the course project for the
IFT6266 class given in Winter 2016.

            Output Size      *   Filter Size   * s_1x1 * e_1x1 * e_3x3 *
            ************************************************************
inputimage: (192, 192,    3) *                 *       *       *       *
conv1:      ( 96,  96,   48) *   7x7/2 (x48)   *       *       *       *
maxpool1:   ( 48,  48,   48) *   3x3/2         *       *       *       *
fire2       ( 48,  48,   64) *                 *   16  *   32  *   32  *
fire3       ( 48,  48,   64) *                 *   16  *   32  *   32  *
fire4       ( 48,  48,  128) *                 *   32  *   64  *   64  *
maxpool4:   ( 24,  24,  128) *   3x3/2         *       *       *       *
fire5       ( 24,  24,  128) *                 *   32  *   64  *   64  *
fire6       ( 24,  24,  192) *                 *   48  *   96  *   96  *
fire7       ( 24,  24,  192) *                 *   48  *   96  *   96  *
fire8       ( 24,  24,  256) *                 *   64  *  128  *  128  *
maxpool8:   ( 12,  12,  256) *   3x3/2         *       *       *       *
fire9       ( 12,  12,  256) *                 *   64  *  128  *  128  *
conv10:     ( 12,  12,    2) *   1x1/1 (x2)    *       *       *       *
avgpool10:  (  1,   1,    2) *   12x12/1       *       *       *       *
softmax:    (  1,   1,    2) *                 *       *       *       *
TOTALS:         NEURONS:                      PARAMETERS:
                                                368386
"""

#
# Imports
#

import ast
import cPickle                              as pkl
import cStringIO
import cv2
import getopt
import gzip
import h5py                                 as H
import inspect
import io
import keras                                as K
import keras.callbacks                      as KC
import keras.layers                         as KL
import keras.layers.advanced_activations    as KLAa
import keras.layers.convolutional           as KLCv
import keras.layers.core                    as KLCo
import keras.layers.normalization           as KLN
import keras.models                         as KM
import keras.optimizers                     as KO
import keras.regularizers                   as KR
import keras.utils.visualize_util           as KUV
import math
import numpy                                as np
import os
import pdb
import pycuda
import pycuda.driver
import pycuda.gpuarray
import pycuda.tools
import socket
import sys
import tarfile
import theano                               as T
import theano.tensor                        as TT
import theano.tensor.nnet                   as TTN
import theano.tensor.nnet.conv              as TTNC
import theano.tensor.nnet.bn                as TTNB
import theano.tensor.signal.pool            as TTSP
from   theano import config                 as TC
import theano.printing                      as TP
import time
import traceback


###############################################################################
# KITNN HDF5 file format
#
# PATH ---------------------------------| TYPE --------------| DESCRIPTION ----
#
# /                                                            Root.
#   sess/                                                      Sessions folder.
#     1/                                                       Session 1 folder.
#       meta/                                                  Metadata folder.
#         initPRNG/                                            Initialization-time MERSENNE TWISTER PRNG state
#           name                          str                  String "MT19937"
#           keys                          uint32[624]          Keys.
#           pos                           uint32               Position in keys
#           has_gauss                     uint32               Have Gaussian?
#           cached_gaussian               float64              Cached Gaussian
#         src.tar.gz                      uint8[]              Source code as of invocation time.
#         argv                            str[]                Arguments as of invocation time.
#         unixTimeStarted                 float64              Invocation time.
#         consistent                      uint64               Is session consistent (all components of "filesystem" created and initialized sanely)?
#       snap/                                                  Snapshots.
#         atomic                          uint64               Atomic toggle indicating if snapshot 0/ or 1/ is the current state.
#         0/                                                   Snapshot 0.
#           data/                                              Training parameters
#             0,1,2,3,4,...               <?>                  Values, velocities and company.
#           ctrl/                                              Control state
#             cc                          str                  Name of continuation function to be called.
#             PRNG/                                            Numpy MERSENNE TWISTER PRNG state
#               name                      str                  String "MT19937"
#               keys                      uint32[624]          Keys.
#               pos                       uint32               Position in keys
#               hasGauss                  uint32               Have Gaussian?
#               gauss                     float64              Cached Gaussian
#             mE                          uint64               Epoch #
#             mTTI                        uint64               Train-over-Training Index
#             mCTI                        uint64               Check-over-Training Index
#             mCVI                        uint64               Check-over-Validation Index
#             mCTErrCnt                   uint64               Errors committed over training set check
#             mCVErrCnt                   uint64               Errors committed over validation set check
#           log/                                               Logging of metrics.
#             trainLoss                   float64[*]           Logging of training loss (NLL).
#             trainErr                    float64[*]           Logging of training error %.
#             validErr                    float64[*]           Logging of validation error %.
#         1/                                                   Snapshot 1.
#           ...                                                Same as 0/
#     2/
#     3/
#     4/
#       ...
#


###############################################################################
# Global constants.
#

H5PY_VLEN_STR           = H.special_dtype(vlen=str)
KITNN_TRAIN_ENTRY_POINT = "KTPrologue"



###############################################################################
# Dummy object class
#

class Object(object): pass


###############################################################################
# Utilities
#

def KXFlush(x):
	"""Flush changes made to a file, group or dataset to disk."""
	x.file.flush()
def KDWritePRNG(group, data):
	"""Write Numpy MT PRNG state to a given HDF5 group."""
	KFDeletePaths(group, ["name", "keys", "pos", "hasGauss", "gauss"])
	
	group.create_dataset("name",            data=data[0], dtype=H5PY_VLEN_STR)
	group.create_dataset("keys",            data=data[1])
	group.create_dataset("pos",             data=data[2])
	group.create_dataset("hasGauss",        data=data[3])
	group.create_dataset("gauss",           data=data[4])
def KDReadPRNG(group):
	"""Read Numpy MT PRNG state from a given HDF5 group."""
	name            = str     (group["name"           ][()])
	keys            = np.array(group["keys"           ][...])
	pos             = int     (group["pos"            ][()])
	has_gauss       = int     (group["hasGauss"       ][()])
	cached_gaussian = float   (group["gauss"          ][()])
	
	return (name, keys, pos, has_gauss, cached_gaussian)
def KFSrcTarGz():
	"""Gzip own source code and return as np.array(, dtype="uint8") ."""
	# Get coordinates of files to crush
	kitnnSrcBase = os.path.dirname(os.path.abspath(__file__))
	sourceFiles = ["kitnn.py", "inpainter.cpp", "imgserver.cpp"]
	
	# Make a magic, in-memory tarfile and write into it all sources.
	f  = cStringIO.StringIO()
	tf = tarfile.open(fileobj=f, mode="w:gz")
	for sources in sourceFiles:
		tf.add(os.path.join(kitnnSrcBase,sources))
	tf.close()
	
	# Get the compressed in-memory tarfile as a byte array.-
	tarGzSource = np.array(bytearray(f.getvalue()), dtype="uint8")
	f.close()
	
	return tarGzSource
def KFDeletePaths(h5pyFileOrGroup, pathList):
	"""Delete unconditionally a path from the file, whether or not it previously existed."""
	if(type(pathList) == str): pathList = [pathList]
	
	for p in pathList:
		if p in h5pyFileOrGroup:
			del h5pyFileOrGroup[p]
def KNCreateSock():
	return socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_IP)
def KNConnectSock(sock, port=5555):
	sock.connect(("127.0.0.1", port))
	return sock
def KNMakePacket(req, batchSize, first, last, sizeIn, sizeOut, s0, s1, maxT, maxR, minS, maxS):
    """Craft a network packet.
    
    req:       0 if random, 1 if sequential
    batchSize: Even, greater than 0
    first:     Smallest index to draw from
    last:      Largest  index to draw from
    sizeIn:    Dataset to warp from. Must be 64, 128 or 256.
    sizeOut:   Output image size. Is the width, height and stride of the X tensor.
    s0, s1:    xorshift128+ PRNG state. Must not be zero.
    maxT:      Maximum translation in pixels.
    maxR:      Maximum rotation, in degrees.
    minS:      Minimum scaling factor.
    maxS:      Maximum scaling factor.
    """
    
    """The packet internally resembles the following:
    typedef struct{
    	uint64_t           req;          /* 0x00 */
    	uint64_t           batchSize;    /* 0x08 */
    	uint64_t           first;        /* 0x10 */
    	uint64_t           last;         /* 0x18 */
    	uint64_t           sizeIn;       /* 0x20 */
    	uint64_t           sizeOut;      /* 0x28 */
    	uint64_t           x128ps0;      /* 0x30 */
    	uint64_t           x128ps1;      /* 0x38 */
    	double             maxT;         /* 0x40 */
    	double             maxR;         /* 0x48 */
    	double             minS;         /* 0x50 */
    	double             maxS;         /* 0x58 */
    } CONN_PKT;
    """
    
    A = np.empty((4,), dtype="uint64")
    B = np.empty((4,), dtype="uint64")
    C = np.empty((4,), dtype="float64")
    
    A[0] = req
    A[1] = batchSize
    A[2] = first
    A[3] = last
    
    B[0] = sizeIn
    B[1] = sizeOut
    B[2] = s0
    B[3] = s1
    
    C[0] = maxT
    C[1] = maxR
    C[2] = minS
    C[3] = maxS
    
    pkt  = bytearray()
    pkt += bytearray(A)
    pkt += bytearray(B)
    pkt += bytearray(C)
    
    return pkt
def KNFetchImgs(sock, req, batchSize, first, last, sizeIn, sizeOut, s0, s1, maxT, maxR, minS, maxS):
	X = np.empty((batchSize,3,sizeOut,sizeOut), dtype="float32")
	Y = np.empty((batchSize,2),                 dtype="float32")
	
	pkt = KNMakePacket(req, batchSize, first, last, sizeIn, sizeOut, s0, s1, maxT, maxR, minS, maxS)
	sock.send(pkt)
	if sock.recv_into(X,0,socket.MSG_WAITALL) != np.prod(X.shape)*4:
		return (None, None)
	if sock.recv_into(Y,0,socket.MSG_WAITALL) != np.prod(Y.shape)*4:
		return (None, None)
	
	return (X,Y)
def KHintonTrick(Y, c):
	"""Remaps the batch of one-hot vectors to a batch of less-confident vectors."""
	if c > 0:
		# Want 1-c in the true class and c divided amongst the n-1 other classes.
		trueC  = 1.0-c
		falseC = c/(Y.shape[1]-1)
		
		return Y*(trueC-falseC) + falseC
	else:
		return Y
def KSparsify(a,b,c):
	#for i in xrange(len(data)):
	#	if data[str(i)][...].ndim == 4 and data[str(i+1)][...].ndim == 4 and data[str(i+2)][...].ndim == 4:
	#		if data[str(i)][...].shape[2:] == (3,3):
	#			KSparsify(data[str(i)][...], data[str(i+1)][...], data[str(i+2)][...])
	
	cAbs    = np.abs(c)
	cMedian = np.median(cAbs)
	mask    = cAbs >= cMedian
	return (a*mask, b*mask, c*mask)
def KIResizeToSize(img, sz):
	Wi = img.shape[1]
	Hi = img.shape[0]
	Wo = sz[1]
	Ho = sz[0]
	
	ai = float(Wi)/float(Hi)
	ao = float(Wo)/float(Ho)
	
	if ao>ai:
		# The output is wider-aspect than the input. Resizing will leave black bands left and right.
		img  = cv2.resize(img, (int(Ho*ai), Ho))
		padL = int((Wo - img.shape[1])/2)
		padR = Wo - img.shape[1] - padL
		img  = cv2.copyMakeBorder(img, 0, 0, padL, padR, cv2.BORDER_CONSTANT)
	else:
		# The input is wider-aspect than the output. Resizing will leave black bands top and bottom.
		img = cv2.resize(img, (Wo, int(Ho/ai)))
		padT= int((Ho - img.shape[0])/2)
		padB= Ho - img.shape[0] - padT
		img = cv2.copyMakeBorder(img, padT, padB, 0, 0, cv2.BORDER_CONSTANT)
	return img

###############################################################################
# KITNN file management.
#

class KITNNFile(Object):
	def __init__                (self, f, mode="r"):
		"""
		Initialize KITNN file.
		"""
		
		self.open(f, mode)
	def open                    (self, f, mode="r"):
		"""
		Open KITNN file.
		
		The argument f can either be a file or a path.
		"""
		if type(f)==str:
			f = H.File(f, mode=mode)
		self.f = f
		self.f.require_group("/sess")
		self.flush()
	def isReadOnly              (self):
		return self.f.mode == "r"
	def flush                   (self):
		"""Flush any changes to the KITNN file to disk."""
		
		if not self.isReadOnly():
			self.f.flush()
	def close                   (self):
		self.f.close()
		del self
	def getSessionNames         (self):
		return sorted(self.f["/sess"].keys(), key=int)
	def existsSession           (self, name):
		return ("/sess/"+name) in self.f
	def getSession              (self, name):
		if self.existsSession(name):
			return KITNNSession(self.f["/sess/"+name])
	def delSession              (self, name):
		if not self.isReadOnly() and self.existsSession(name):
			del self.f["/sess/"+name]
	def prune                   (self, verbose=True):
		"""Prune inconsistent sessions from the file."""
		
		if not self.isReadOnly():
			for d in self.getSessionNames():
				if not self.getSession(d).isConsistent():
					if verbose:
						print("Prunning inconsistent session \""+d+"\" ...")
					self.delSession(d)
			self.flush()
		
		return self
	def createSession           (self, name):
		if self.existsSession(name):
			raise KeyError("Session "+name+" already exists!")
		
		if not self.isReadOnly():
			return KITNNSession(self.f.require_group("/sess/"+name), False)
	def getLastConsistentSession(self):
		for s in self.getSessionNames()[::-1]:
			sess = self.getSession(s)
			if sess.isConsistent():
				return sess
	def createNextSession       (self):
		if not self.isReadOnly():
			sess = self.getLastConsistentSession()
			if(sess == None):
				return self.createSession("1")
			else:
				return self.createSession(str(int(sess.getName())+1)).initFromOldSession(sess)

###############################################################################
# KITNN training session management.
#

class KITNNSession(Object):
	def __init__                (self, d, readOnly=True, **kwargs):
		"""Initialize KITNN session object from a given file object and session group."""
		self.__dict__.update(**kwargs)
		self.d            = d
		self.readOnly     = self.d.file.mode == "r" or readOnly
		
		#
		# Initialization work.
		#
		
		if not self.isReadOnly():
			# We have readwrite intent, so we initialize ourselves. We do not, however,
			# mark ourselves consistent, since we're unaware of the requirements of the model just yet.
			self.initMeta()
	def isReadOnly              (self):
		return self.readOnly
	def flush                   (self):
		self.d.file.flush()
	def getName                 (self):
		return os.path.basename(self.d.name)
	def isConsistent            (self):
		if "meta/consistent" not in self.d:
			return False
		consistent = self.d.require_dataset("meta/consistent", shape=(), dtype="uint64", exact=True)
		return consistent[()] == 1
	def markConsistent          (self, verbose=True):
		self.flush()
		self.d.require_dataset("meta/consistent", shape=(), dtype="uint64", exact=True)[()] = 1
		self.flush()
		
		if verbose and not self.isConsistent():
			print("Marked session "+self.getName()+" consistent.")
		
		return self
	def initMeta                (self):
		"""Initialize metadata for current session."""
		KFDeletePaths(self.d, ["meta/initPRNG",
		                       "meta/src.tar.gz",
		                       "meta/argv",
		                       "meta/unixTimeStarted",
		                       "meta/theanoVersion",
		                       "meta/consistent"])
		
		initPRNG        = np.random.get_state()
		tarGzSrc        = KFSrcTarGz()
		unixTimeStarted = np.full((), time.time(), dtype="float64")
		consistent      = np.full((), 0, dtype="uint64")
		theanoVersion   = T.version.full_version
		
		KDWritePRNG(self.d.require_group("meta/initPRNG"), data=initPRNG)
		self.d.create_dataset("meta/src.tar.gz",           data=tarGzSrc)
		self.d.create_dataset("meta/argv",                 data=sys.argv, dtype=H5PY_VLEN_STR)
		self.d.create_dataset("meta/theanoVersion",        data=theanoVersion, dtype=H5PY_VLEN_STR)
		self.d.create_dataset("meta/unixTimeStarted",      data=unixTimeStarted)
		self.d.create_dataset("meta/consistent",           data=consistent)
	def initFromOldSession      (self, oldSess):
		"""Initialize on-disk snapshots from old session."""
		assert(type(oldSess) == KITNNSession)
		assert(oldSess.isConsistent())
		
		oldSess.d.copy("snap", self.d)
		self.markConsistent()
		
		return self
	def getTrainer              (self, *args, **kwargs):
		return KITNNTrainer(self, *args, **kwargs)


###############################################################################
# KITNN Trainer class code.
#

class KITNNTrainer(Object):
	def __init__(self, sess = None, *args, **kwargs):
		"""Construct a trainer object from a session."""
		
		# Initialize a few constants to default values.
		self.kTB           = 50
		self.kCB           = 250
		
		# Initialize the mutable state of the trainer.
		self.cc            = KITNN_TRAIN_ENTRY_POINT
		self.mC            = eval(self.cc)
		self.mE            = 0
		self.mTTI          = 0
		self.mCTI          = 0
		self.mCVI          = 0
		self.mCTErrCnt     = 0xFFFFFFFFFFFFFFFF
		self.mCVErrCnt     = 0xFFFFFFFFFFFFFFFF
		self.logTrainLoss  = []
		self.logTrainErr   = []
		self.logValidErr   = []
		
		
		# Load parameters
		self.model         = self.constructModel(*args, **kwargs)
		if sess != None and type(sess) == KITNNSession:
			self.sess = sess
			if self.sess.isConsistent():
				self.load()
			elif not self.sess.readOnly:
				self.save()
				self.sess.markConsistent()
			else:
				raise Exception("Inconsistent and read-only session!")
		else:
			self.sess = None
	def constructModel(self, *args, **kwargs):
		print("Model Compilation Start...")
		
		############## Initialization ################
		convInit = "he_normal"
		
		############## Regularization ################
		reg      = KR.l1l2(l1=0.0002, l2=0.0002)
		#reg      = None
		
		############## Optimizer      ################
		baseLr = 0.001
		#opt      = KO.SGD     (lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
		#opt      = KO.RMSprop ()
		#opt      = KO.Adagrad ()
		#opt      = KO.Adadelta()
		opt      = KO.Adam    (lr=baseLr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipvalue=1)
		#opt      = KO.Adamax  (lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
		
		opt.baseLr  = baseLr
		opt.lrDecay = 0.97
		
		############### Structural #################
		f1     =  96;
		f2_s1  =  32; f2_e1 = 128; f2_e3 = 128;
		f3_s1  =  32; f3_e1 = 128; f3_e3 = 128;
		f4_s1  =  32; f4_e1 = 128; f4_e3 = 128;
		f5_s1  =  32; f5_e1 = 128; f5_e3 = 128;
		f6_s1  =  48; f6_e1 = 192; f6_e3 = 192;
		f7_s1  =  48; f7_e1 = 192; f7_e3 = 192;
		f8_s1  =  64; f8_e1 = 256; f8_e3 = 256;
		f9_s1  =  64; f9_e1 = 256; f9_e3 = 256;
		f10    =   2;
		
		############### Model         #################
		model = KM.Graph()
		
		model.add_input ("input",  (3,192,192), (50,3,192,192))
		model.add_node  (KLCv.AveragePooling2D  ((2,2), (2,2), "valid"),                              "input_medium",       input="input")
		model.add_node  (KLCv.AveragePooling2D  ((2,2), (2,2), "valid"),                              "input_coarse",       input="input_medium")
		
		#model.add_node  (KLCo.Dropout           (0.5),                                                "dropout_input_fine",   input="input")
		#model.add_node  (KLCo.Dropout           (0.5),                                                "dropout_input_medium", input="input_medium")
		#model.add_node  (KLCo.Dropout           (0.5),                                                "dropout_input_coarse", input="input_coarse")
		
		model.add_node  (KLCv.Convolution2D     (f1,    7, 7, border_mode="same", subsample=(1,1), init=convInit, W_regularizer=reg),   "conv1_fine/act",     input="input")
		model.add_node  (KLCv.Convolution2D     (f1,    7, 7, border_mode="same", subsample=(1,1), init=convInit, W_regularizer=reg),   "conv1_medium/act",   input="input_medium")
		model.add_node  (KLCv.Convolution2D     (f1,    7, 7, border_mode="same", subsample=(1,1), init=convInit, W_regularizer=reg),   "conv1_coarse/act",   input="input_coarse")
		
		model.add_node  (KLCv.MaxPooling2D      ((4,4), (4,4), "same"),                               "finepool/out",       input="conv1_fine/act")
		model.add_node  (KLCv.MaxPooling2D      ((2,2), (2,2), "same"),                               "mediumpool/out",     input="conv1_medium/act")
		
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn1/out",            inputs=["finepool/out", "mediumpool/out", "conv1_coarse/act"], concat_axis=1)
		
		model.add_node  (KLCo.Activation        ("relu"),                                             "conv1/out",          input="bn1/out")
		
		model.add_node  (KLCv.Convolution2D     (f2_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire2/comp/act",     input="conv1/out")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn2c/out",           input="fire2/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire2/comp/out",     input="bn2c/out")
		model.add_node  (KLCv.Convolution2D     (f2_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire2/exp1/act",     input="fire2/comp/out")
		model.add_node  (KLCv.Convolution2D     (f2_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire2/exp3/act",     input="fire2/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire2/exp/out",      inputs=["fire2/exp1/act", "fire2/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCv.Convolution2D     (f3_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire3/comp/act",     input="fire2/exp/out")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn3c/out",           input="fire3/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire3/comp/out",     input="bn3c/out")
		model.add_node  (KLCv.Convolution2D     (f3_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire3/exp1/act",     input="fire3/comp/out")
		model.add_node  (KLCv.Convolution2D     (f3_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire3/exp3/act",     input="fire3/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire3/exp/out",      inputs=["fire3/exp1/act", "fire3/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCv.Convolution2D     (f4_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire4/comp/act",     inputs=["fire2/exp/out", "fire3/exp/out"], merge_mode="sum")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn4c/out",           input="fire4/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire4/comp/out",     input="bn4c/out")
		model.add_node  (KLCv.Convolution2D     (f4_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire4/exp1/act",     input="fire4/comp/out")
		model.add_node  (KLCv.Convolution2D     (f4_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire4/exp3/act",     input="fire4/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire4/exp/out",      inputs=["fire4/exp1/act", "fire4/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCv.MaxPooling2D      ((3,3), (2,2), "same"),                               "maxpool4/out",       input="fire4/exp/out")
		
		model.add_node  (KLCv.Convolution2D     (f5_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire5/comp/act",     input="maxpool4/out")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn5c/out",           input="fire5/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire5/comp/out",     input="bn5c/out")
		model.add_node  (KLCv.Convolution2D     (f5_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire5/exp1/act",     input="fire5/comp/out")
		model.add_node  (KLCv.Convolution2D     (f5_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire5/exp3/act",     input="fire5/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire5/exp/out",      inputs=["fire5/exp1/act", "fire5/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCv.Convolution2D     (f6_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire6/comp/act",     inputs=["maxpool4/out", "fire5/exp/out"], merge_mode="sum")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn6c/out",           input="fire6/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire6/comp/out",     input="bn6c/out")
		model.add_node  (KLCv.Convolution2D     (f6_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire6/exp1/act",     input="fire6/comp/out")
		model.add_node  (KLCv.Convolution2D     (f6_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire6/exp3/act",     input="fire6/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire6/exp/out",      inputs=["fire6/exp1/act", "fire6/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCv.Convolution2D     (f7_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire7/comp/act",     input="fire6/exp/out")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn7c/out",           input="fire7/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire7/comp/out",     input="bn7c/out")
		model.add_node  (KLCv.Convolution2D     (f7_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire7/exp1/act",     input="fire7/comp/out")
		model.add_node  (KLCv.Convolution2D     (f7_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire7/exp3/act",     input="fire7/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire7/exp/out",      inputs=["fire7/exp1/act", "fire7/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCv.Convolution2D     (f8_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire8/comp/act",     inputs=["fire6/exp/out", "fire7/exp/out"], merge_mode="sum")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn8c/out",           input="fire8/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire8/comp/out",     input="bn8c/out")
		model.add_node  (KLCv.Convolution2D     (f8_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire8/exp1/act",     input="fire8/comp/out")
		model.add_node  (KLCv.Convolution2D     (f8_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire8/exp3/act",     input="fire8/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire8/exp/out",      inputs=["fire8/exp1/act", "fire8/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCv.MaxPooling2D      ((3,3), (2,2), "same"),                               "maxpool8/out",       input="fire8/exp/out")
		
		model.add_node  (KLCv.Convolution2D     (f9_s1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire9/comp/act",     input="maxpool8/out")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn9c/out",           input="fire9/comp/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire9/comp/out",     input="bn9c/out")
		model.add_node  (KLCv.Convolution2D     (f9_e1, 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "fire9/exp1/act",     input="fire9/comp/out")
		model.add_node  (KLCv.Convolution2D     (f9_e3, 3, 3, border_mode="same", init=convInit, W_regularizer=reg),     "fire9/exp3/act",     input="fire9/comp/out")
		model.add_node  (KLCo.Activation        ("relu"),                                             "fire9/exp/out",      inputs=["fire9/exp1/act", "fire9/exp3/act"], concat_axis=1)
		
		model.add_node  (KLCo.Dropout           (0.5),                                                "dropout9/out",       inputs=["maxpool8/out", "fire9/exp/out"], merge_mode="sum")
		
		model.add_node  (KLCv.Convolution2D     (f10  , 1, 1, border_mode="same", init=convInit, W_regularizer=reg),     "conv10/act",         input="dropout9/out")
		model.add_node  (KLN. BatchNormalization(axis=1),                                             "bn10/out",           input="conv10/act")
		model.add_node  (KLCo.Activation        ("relu"),                                             "conv10/out",         input="bn10/out")
		
		model.add_node  (KLCv.AveragePooling2D  ((12,12), (1,1), "valid"),                            "avgpool10/out",      input="conv10/out")
		
		model.add_node  (KLCo.Reshape           ((f10,)),                                             "softmax/in",         input="avgpool10/out")
		model.add_node  (KLCo.Activation        ("softmax"),                                          "softmax/out",        input="softmax/in")
		
		model.add_output("output", "softmax/out")
		
		model.compile(loss={"output":'categorical_crossentropy'}, optimizer=opt)
		#model.powerf = T.function()
		KUV.plot(model, to_file='model.png', show_shape=True)
		
		print("Model Compilation End.")
		
		#pdb.set_trace()
		
		return model
	def load(self):
		"""Load state from the "current" snapshot."""
		
		if not self.haveSession():
			return
		
		# Get current snapshot.
		snap = self.getCurrSnap()
		
		# Read all the state.
		np.random.set_state(KDReadPRNG(snap["ctrl/PRNG"]))
		self.cc            = str(snap["ctrl/cc"][()])
		self.mC            = eval(self.cc)
		self.mE            = long(snap["ctrl/mE"][()])
		self.mTTI          = long(snap["ctrl/mTTI"][()])
		self.mCTI          = long(snap["ctrl/mCTI"][()])
		self.mCVI          = long(snap["ctrl/mCVI"][()])
		self.mCTErrCnt     = long(snap["ctrl/mCTErrCnt"][()])
		self.mCVErrCnt     = long(snap["ctrl/mCVErrCnt"][()])
		self.logTrainLoss  = snap["log/trainLoss"][...].tolist()
		self.logTrainErr   = snap["log/trainErr"][...].tolist()
		self.logValidErr   = snap["log/validErr"][...].tolist()
		optState = []
		for dataset in sorted(snap["data/"].keys(), key=int):
			optState.append(snap["data/"+dataset][...])
		self.model.optimizer.set_state(optState)
		self.updateLR()
	def save(self):
		"""Save state to the "next" snapshot.
		Then, flip the buffer atomically."""
		
		if not self.haveSession():
			return
		
		# Get next snapshot.
		snap = self.getNextSnap()
		
		# Write all the state.
		# ctrl/
		KDWritePRNG(snap.require_group("ctrl/PRNG"), np.random.get_state())
		snap.require_dataset("ctrl/cc",        (), H5PY_VLEN_STR, exact=True)[()] = str        (self.cc)
		snap.require_dataset("ctrl/mE",        (), "uint64",      exact=True)[()] = int        (self.mE)
		snap.require_dataset("ctrl/mTTI",      (), "uint64",      exact=True)[()] = int        (self.mTTI)
		snap.require_dataset("ctrl/mCTI",      (), "uint64",      exact=True)[()] = int        (self.mCTI)
		snap.require_dataset("ctrl/mCVI",      (), "uint64",      exact=True)[()] = int        (self.mCVI)
		snap.require_dataset("ctrl/mCTErrCnt", (), "uint64",      exact=True)[()] = int        (self.mCTErrCnt)
		snap.require_dataset("ctrl/mCVErrCnt", (), "uint64",      exact=True)[()] = int        (self.mCVErrCnt)
		# log/
		if "log/trainLoss" not in snap:
			snap.require_dataset("log/trainLoss", (0,), "float64", maxshape=(None,))
		if "log/trainErr"  not in snap:
			snap.require_dataset("log/trainErr",  (0,), "float64", maxshape=(None,))
		if "log/validErr"  not in snap:
			snap.require_dataset("log/validErr",  (0,), "float64", maxshape=(None,))
		snap["log/trainLoss"].resize((len(self.logTrainLoss),))
		snap["log/trainLoss"][...]   = np.array(self.logTrainLoss)
		snap["log/trainErr"] .resize((len(self.logTrainErr),))
		snap["log/trainErr"] [...]   = np.array(self.logTrainErr)
		snap["log/validErr"] .resize((len(self.logValidErr),))
		snap["log/validErr"] [...]   = np.array(self.logValidErr)
		# data/
		optState = self.model.optimizer.get_state()
		for i in xrange(len(optState)):
			s = optState[i]
			snap.require_dataset("data/"+str(i), s.shape, s.dtype, exact=True)[...] = s
		
		# Flip the snapshots atomically.
		self.toggleSnapNum()
	def updateLR(self):
		self.model.optimizer.lr.set_value(self.model.optimizer.baseLr * (self.model.optimizer.lrDecay ** self.mE))
	def train(self):
		"""Train a KITNN.
		
		This method assumes that the trainer object is fully initialized, and in
		particular that the present continuation is in self.mC.
		
		It also assumes an image server is running on the default port."""
		
		self.sock = KNConnectSock(KNCreateSock())
		
		try:
			while callable(self.mC):
				self.mC = self.mC(self)
		except:
			traceback.print_exc()
			pdb.set_trace()
		finally:
			print("\nStopped.")
			return self.mC
	def invoke(self, mC, snap=False, newLine=False, **kwargs):
		"""Invoke a continuation, possibly snapshotting and printing to stdout as well."""
		
		if callable(snap): snap=bool(snap())
		
		#
		# (Maybe) take a snapshot. We make a commitment to call mC, then
		# immediately invoke it. There must be **NO** state-changing after the
		# end of this "if" and before the calling of committed continuation.
		#
		
		self.cc = mC.func_name
		if snap:
			self.save()
			
			sys.stdout.write("  Snapshot!")
			sys.stdout.flush()
		
		return mC
	def shouldTTSnap(self):
		return False
	def shouldCTSnap(self):
		return False
	def shouldCVSnap(self):
		return False
	def log(self, logEntries):
		"""Logger."""
		
		if "trainLoss" in logEntries:
			self.logTrainLoss.append(float(logEntries["trainLoss"]))
		if "trainErr" in logEntries:
			self.logTrainErr.append(float(logEntries["trainErr"]))
		if "validErr" in logEntries:
			self.logValidErr.append(float(logEntries["validErr"]))
	def haveSession             (self):
		return self.sess != None
	def getCurrSnapNum          (self):
		if "snap/atomic" not in self.sess.d:
			self.sess.d.create_dataset("snap/atomic", data=np.full((), 0, dtype="uint64"))
		return int(self.sess.d["snap/atomic"][()])
	def getCurrSnap             (self):
		if "snap/atomic" not in self.sess.d:
			self.sess.d.create_dataset("snap/atomic", data=np.full((), 0, dtype="uint64"))
		return self.sess.d.require_group("snap/"+str(self.getCurrSnapNum()))
	def getNextSnapNum          (self):
		if "snap/atomic" not in self.sess.d:
			self.sess.d.create_dataset("snap/atomic", data=np.full((), 0, dtype="uint64"))
		return int(self.sess.d["snap/atomic"][()])^1
	def getNextSnap             (self):
		if "snap/atomic" not in self.sess.d:
			self.sess.d.create_dataset("snap/atomic", data=np.full((), 0, dtype="uint64"))
		return self.sess.d.require_group("snap/"+str(self.getNextSnapNum()))
	def toggleSnapNum           (self):
		if "snap/atomic" not in self.sess.d:
			self.sess.d.create_dataset("snap/atomic", data=np.full((), 0, dtype="uint64"))
		self.sess.flush()
		self.sess.d["snap/atomic"][()] = self.getNextSnapNum()
		self.sess.flush()
	def classify                (self, imgs):
		"""Run classification on (B,3,H,W) tensor of images using current snapshot.
		Returns a (B,2) tensor of probabilities."""
		
		return graph.predict({'input':imgs})["output"]



###############################################################################
# KITNN Trainer Training Loop Code.
#

def cps(f):
	f.isCps = 1
	return f
@cps
def KTPrologue(cc):
	cc.mE = 0
	return cc.invoke(KTEpochLoopStart)
@cps
def KTEpochLoopStart(cc):
	#
	# Epoch Loop CONDITION
	#
	if(True):
		#
		# Epoch Loop BODY
		#
		
		# Progress indexes
		cc.mTTI      = 0
		cc.mCTI      = 0
		cc.mCVI      = 0
		cc.mCTErrCnt = 0
		cc.mCVErrCnt = 0
		
		return cc.invoke(KTTrainOverTrainLoop)
	else:
		#
		# Epoch Loop EPILOGUE
		#
		
		#UNREACHABLE
		pass
@cps
def KTTrainOverTrainLoop(cc):
	#
	# Train-over-Train Loop CONDITION
	#
	if(cc.mTTI + cc.kTB <= 22500):
		#
		# Train-over-Train Loop BODY
		#
		
		ts = time.time()
		
		s  = np.random.randint(1, 2**62, (2,)).astype("uint64")
		#                  Socket,  Rq#, B,      first,   last,  sizeIn, sizeOut, x128+s0, x128+s1, maxT, maxR, minS, maxS
		X, Y = KNFetchImgs(cc.sock, 0,   cc.kTB, 0,       22500, 256,    192,     s[0],    s[1],    16,   60,   0.8,  1.2)
		Y = KHintonTrick(Y, c=0)
		
		tm = time.time()
		
		loss = cc.model.train_on_batch({"input":X, "output":Y})
		
		cc.mTTI += cc.kTB
		cc.log({"trainLoss":float(loss[0][()]/np.log(2))})
		
		te = time.time()
		
		sys.stdout.write("\nEpoch: {:4d}  Iter {:4d}  Loss: {:20.17f}  Time: {:.4f}+{:.4f}s".format(cc.mE, cc.mTTI/cc.kTB, loss[0][()]/np.log(2), tm-ts, te-tm))
		sys.stdout.flush()
		
		return cc.invoke(KTTrainOverTrainLoop, snap=cc.shouldTTSnap)
	else:
		#
		# Train-over-Train Loop EPILOGUE
		#
		sys.stdout.write("\n")
		sys.stdout.flush()
		return cc.invoke(KTCheckOverTrainLoop, snap=True)
@cps
def KTCheckOverTrainLoop(cc):
	#
	# Check-over-Train Loop CONDITION
	#
	if(cc.mCTI + cc.kCB <= 22500):
		#
		# Check-over-Train Loop BODY
		#
		
		#                  Socket,  Rq#, B,      first,   last,           sizeIn, sizeOut, x128+s0, x128+s1, maxT, maxR, minS, maxS
		X, Y = KNFetchImgs(cc.sock, 1,   cc.kCB, cc.mCTI, cc.mCTI+cc.kCB, 256,    192,     1,       1,       0,    0,    1.0,  1.0)
		YEst   = cc.model.predict({"input":X})["output"]
		yDiff  = np.argmax(Y, axis=1) != np.argmax(YEst, axis=1)
		
		cc.mCTI      += cc.kCB
		cc.mCTErrCnt += long(np.sum(yDiff))
		
		sys.stdout.write("\rChecking... {:5d} train set errors on {:5d} checked ({:7.3f}%)".format(
		                 cc.mCTErrCnt, cc.mCTI, 100.0*float(cc.mCTErrCnt)/cc.mCTI))
		sys.stdout.flush()
		
		return cc.invoke(KTCheckOverTrainLoop, snap=cc.shouldCTSnap)
	else:
		#
		# Check-over-Train Loop EPILOGUE
		#
		
		cc.log({"trainErr":float(cc.mCTErrCnt)/cc.mCTI})
		sys.stdout.write("\n")
		sys.stdout.flush()
		return cc.invoke(KTCheckOverValidLoop, snap=True)
@cps
def KTCheckOverValidLoop(cc):
	#
	# Check-over-Valid Loop CONDITION
	#
	if(cc.mCVI + cc.kCB <= 2500):
		#
		# Check-over-Valid Loop BODY
		#
		
		#                  Socket,  Rq#, B,      first,         last,                 sizeIn, sizeOut, x128+s0, x128+s1, maxT, maxR, minS, maxS
		X, Y = KNFetchImgs(cc.sock, 1,   cc.kCB, 22500+cc.mCVI, 22500+cc.mCVI+cc.kCB, 256,    192,     1,       1,       0,    0,    1.0,  1.0)
		YEst   = cc.model.predict({"input":X})["output"]
		
		yDiff  = np.argmax(Y, axis=1) != np.argmax(YEst, axis=1)
		
		cc.mCVI      += cc.kCB
		cc.mCVErrCnt += long(np.sum(yDiff))
		
		sys.stdout.write("\rChecking... {:5d} valid set errors on {:5d} checked ({:7.3f}%)".format(
		                 cc.mCVErrCnt, cc.mCVI, 100.0*float(cc.mCVErrCnt)/cc.mCVI))
		sys.stdout.flush()
		
		return cc.invoke(KTCheckOverValidLoop, snap=cc.shouldCVSnap)
	else:
		#
		# Check-over-Valid Loop EPILOGUE
		#
		
		cc.log({"validErr":float(cc.mCVErrCnt)/cc.mCVI})
		sys.stdout.write("\n")
		sys.stdout.flush()
		return cc.invoke(KTEpochLoopEnd, snap=False)
@cps
def KTEpochLoopEnd(cc):
	# Save if best model so far.
	#cc.saveIfBestSoFar()
	
	# Increment epoch number
	cc.mE += 1
	cc.updateLR()
	
	# Loop
	return cc.invoke(KTEpochLoopStart, snap=True, newLine=True)





###############################################################################
# Implementations of the script's "verbs".
#

#
# Print help/usage information
#

def verb_help(argv=None):
	print("""
Usage of KITNN.

The KITNN script is invoked using a verb that denotes the general action to be
taken, plus optional arguments. The following verbs are defined:

    \033[1mclassify\033[0m:

    \033[1mhelp\033[0m:
        This help message.

    \033[1mtrain\033[0m:

"""[1:-1] #This hack deletes the newlines before and after the triple quotes.
	)
def verb_classify(argv):
	"""Classify the argument images as cat or dog."""
	
	print argv
def verb_train(argv=None):
	"""Train KITNN."""
	
	# Open session file, clean out inconsistent sessions and create a consistent session.
	# Then, run training with this consistent session.
	KITNNFile(argv[2], "a").prune().createNextSession().getTrainer().train()
	
	# UNREACHABLE, because training is (should be) an infinite loop.
	pdb.set_trace()
def verb_screw(argv=None):
	"""Screw around."""
	
	pass
def verb_testimgserver(argv=None):
	"""Test Image Server"""
	
	sock  = KNConnectSock(KNCreateSock())
	(X,Y) = KNFetchImgs(sock, 0, 50, 0, 50, 256, 192, 1234, 2345, 16, 60, 0.6, 1.2)
	
	x = np.einsum("Bchw->Bhwc", X)
	y = Y
	
	for xi in x:
		print np.sum(np.abs(xi))
		cv2.imshow("Warped", xi)
		cv2.waitKey()
	
	#pdb.set_trace()
def verb_create(argv):
	"""Create session."""
	
	name = argv[2]
	
	# Guard against the file already existing
	if os.path.isfile(name):
		print("ERROR: File \""+name+"\" already exists!")
		return
	
	# Create file.
	KITNNFile(name, "a").createNextSession().getTrainer()
def verb_interactive(argv=None):
	"""Interactive mode."""
	
	pdb.set_trace()
def verb_buildaux(argv=None):
	"""Build auxiliary applications."""
	
	import subprocess
	
	cmds = [
	    """g++ -O3 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_photo -lhdf5 -fopenmp inpainter.cpp -o inpainter""",
	    """g++ -O3 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lhdf5 -fopenmp imgserver.cpp -o imgserver"""
	]
	
	try:
		for cmd in cmds:
			sys.stdout.write(subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True))
	except subprocess.CalledProcessError as cpe:
		sys.stdout.write(cpe.output)
def verb_summary(argv=None):
	"Dump a summary of the network."
	trainer = KITNNTrainer()
	trainer.model.summary()
	KUV.plot(trainer.model, to_file='model.png', show_shape=True)
def verb_showfilters(argv):
	"""Dump source code of session"""
	
	f  = KITNNFile(argv[2])
	s  = f.getSession(argv[3]).d["snap/1"]
	
	fine   = s["data/2" ][...]
	medium = s["data/8" ][...]
	coarse = s["data/14"][...]
	
	w  = 3+(16*7+15*3)+3
	h  = 3+( 9*7+ 8*3)+3
	
	img= np.zeros((h,w,3), dtype="uint8")
	
	for i in xrange(9):
		for j in xrange(16):
			n     = i*16+j
			if i in [0,1,2]:
				card  = fine  [n- 0]
			elif i in [3,4,5]:
				card  = medium[n-48]
			elif i in [6,7,8]:
				card  = coarse[n-96]
			card -= np.min(card)
			card /= np.max(card)
			card  = card.transpose(1,2,0)
			
			img[3+i*10:3+i*10+7, 3+j*10:3+j*10+7] = 255*card
	
	img = cv2.resize(img, (0,0), None, 8, 8, cv2.INTER_NEAREST)
	cv2.imshow("Filters", img)
	cv2.imwrite("Filters.png", img)
	cv2.waitKey()
def verb_dumpsrcs(argv):
	"""Dump source code of session"""
	
	f  = KITNNFile(argv[2])
	gz = bytearray(f.getSession(argv[3]).d["meta/src.tar.gz"])
	sys.stdout.write(gz)
	f.close()
def verb_dumpargs(argv):
	"""Dump arguments of session"""
	
	f  = KITNNFile(argv[2])
	print list(f.getSession(argv[3]).d["meta/argv"][...])
	f.close()
def verb_test(argv):
	"""Generate test set"""
	f       = KITNNFile(argv[2])
	sess    = f.getLastConsistentSession()
	trainer = sess.getTrainer()
	
	print "id,label"
	
	# Loop over test set
	testPath = argv[3]
	for i in xrange(12500/trainer.kCB):
		# Allocate memory
		X = np.empty((trainer.kCB, 3, 192, 192), dtype="float32")
		
		# Resize images
		for j in xrange(trainer.kCB):
			n   = trainer.kCB*i + j + 1
			img = cv2.imread(os.path.join(testPath, str(n)+".jpg"))
			img = KIResizeToSize(img, (192,192))
			X[j] = img.transpose(2,0,1).astype("float32")
		
		# Recognize
		Y = np.argmax(trainer.classify(X*2.0/255.0 - 1.0), axis=1)
		
		# Print results
		for j in xrange(trainer.kCB):
			n = trainer.kCB*i + j + 1
			print str(n)+","+str(Y[j])


###############################################################################
# Main
#

if __name__ == "__main__":
	#
	# This script is invoked using a verb that denotes the general action to
	# take, plus optional arguments. If no verb is provided or the one
	# provided does not match anything, print a help message.
	#
	
	if((len(sys.argv) > 1)                      and # Have a verb?
	   ("verb_"+sys.argv[1] in globals())       and # Have symbol w/ this name?
	   (callable(eval("verb_"+sys.argv[1])))):      # And it is callable?
		eval("verb_"+sys.argv[1]+"(sys.argv)")      # Then call it.
	else:
		verb_help(sys.argv)                         # Or offer help.
