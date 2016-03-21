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
import cPickle as pkl
import cStringIO
import cv2
import getopt
import gzip
import h5py                            as H
import inspect
import io
import math
import numpy as np
import os
import pdb
import sys
import tarfile
import theano                          as T
import theano.tensor                   as TT
import theano.tensor.nnet              as TTN
import theano.tensor.nnet.conv         as TTNC
import theano.tensor.signal.pool       as TTSP
from   theano import config            as TC
import theano.printing                 as TP
import time


###############################################################################
# KITNN HDF5 file format
#
# PATH ---------------------------------| TYPE --------------| DESCRIPTION ----
#
# /                                                            Root.
#   sessions/                                                  Sessions folder.
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
#       snapshot/                                              Snapshots.
#         atomic                          uint64               Atomic toggle indicating if snapshot 0/ or 1/ is the current state.
#         0/                                                   Snapshot 0.
#           currKITNN/                                         Current KITNN.
#             data/                                            Training 
#               parameters/                                    Parameters.
#                 <paramHierarchy>        T                    *** MODEL-DEPENDENT ***
#               velocities/                                    Velocities for momentum methods
#                 <paramHierarchy>        T                    *** MODEL-DEPENDENT ***
#             misc/                                            Miscellaneous state
#               mC                        str                  Name of continuation function to be called.
#               PRNG/                                          Numpy MERSENNE TWISTER PRNG state
#                 name                    str                  String "MT19937"
#                 keys                    uint32[624]          Keys.
#                 pos                     uint32               Position in keys
#                 has_gauss               uint32               Have Gaussian?
#                 cached_gaussian         float64              Cached Gaussian
#               mE                        uint64               Epoch #
#               mTTI                      uint64               Train-over-Training Index
#               mCTI                      uint64               Check-over-Training Index
#               mCVI                      uint64               Check-over-Validation Index
#               mCTErrCnt                 uint64               Errors committed over training set check
#               mCVErrCnt                 uint64               Errors committed over validation set check
#             log/                                             Logging of metrics.
#               trainLoss                 float64[*]           Logging of training loss (NLL).
#               trainErr                  float64[*]           Logging of training error %.
#               validErr                  float64[*]           Logging of validation error %.
#           bestKITNN/                                         Best KITNN so far.
#             ...                                              Same as currKITNN/
#         1/                                                   Snapshot 1.
#           ...                                                Same as 0/
#     2/
#     3/
#     4/
#       ...
#

###############################################################################
# KITNN training process PSEUDOCODE!!!!
def kitnnTrain():
	#
	# READ-ONLY VARS:
	# - kTB                                  Training Batch Size.
	# - kCB                                  Check    Batch Size.
	#
	# READ-WRITE VARS:
	# - <Numpy MT PRNG state>
	# - mE                                   Epoch #
	# - mTTI                                 Train-over-Training Index
	# - mCTI                                 Check-over-Training Index
	# - mCVI                                 Check-over-Validation Index
	# - mCTErrCnt                            Errors committed over training set check
	# - mCVErrCnt                            Errors committed over validation set check
	# - mCVBestErrCnt                        Errors committed by best model so far in check over validation set
	#
	
	self.mE = 0
	while True: #For each epoch,
		# Progress indexes
		self.mTTI = 0
		self.mCTI = 0
		self.mCVI = 0
		self.printStatus(snap=False)
		
		# Train over Training Set
		while(self.mTTI + kTB < NUM_TRAIN):
			self.uploadTrainData(self.mTTI, kCB)
			loss = self.invokeTrainF()
			
			self.mTTI += kTB
			self.log({"trainLoss":float(loss)})
			
			self.printStatus(snap=self.shouldTTSnap)
		self.printStatus(snap=True)
		
		# Check over Training Set
		while(self.mCTI + kCB < NUM_TRAIN):
			self.uploadTrainData(self.mCTI, kCB)
			yEst = self.invokeClassF()
			
			self.mCTI      += kCB
			self.mCTErrCnt += np.sum(np.argmax(yTrue, axis=1) != np.argmax(yEst))
			
			self.printStatus(snap=self.shouldCTSnap)
		self.log({"trainErr":float(self.mCTErrCnt)/self.mCTI})
		self.printStatus(snap=True)
		
		# Check over Validation Set
		while(self.mCVI + kCB < NUM_VALID):
			self.uploadValidData(self.mCVI, kCB)
			yEst = self.invokeClassF()
			
			self.mCVI      += kCB
			self.mCVErrCnt += np.sum(np.argmax(yTrue, axis=1) != np.argmax(yEst))
			
			self.printStatus(snap=self.shouldCVSnap)
		self.log({"validErr":float(self.mCVErrCnt)/self.mCVI})
		self.printStatus(snap=False)
		
		# Save if best model so far.
		self.saveIfBestSoFar()
		
		#Increment epoch number
		self.mE += 1
		self.printStatus(snap=True, newLine=True)


###############################################################################
# Global constants.
#

H5PY_VLEN_STR = H.special_dtype(vlen=str)

#
# SqueezeNet configuration
#

conv1  =  48;
f2_s1  =  16; f2_e1 =  32; f2_e3 =  32; f2_e = f2_e1 + f2_e3;
f3_s1  =  16; f3_e1 =  32; f3_e3 =  32; f3_e = f3_e1 + f3_e3;
f4_s1  =  32; f4_e1 =  64; f4_e3 =  64; f4_e = f4_e1 + f4_e3;
f5_s1  =  32; f5_e1 =  64; f5_e3 =  64; f5_e = f5_e1 + f5_e3;
f6_s1  =  48; f6_e1 =  96; f6_e3 =  96; f6_e = f6_e1 + f6_e3;
f7_s1  =  48; f7_e1 =  96; f7_e3 =  96; f7_e = f7_e1 + f7_e3;
f8_s1  =  64; f8_e1 = 128; f8_e3 = 128; f8_e = f8_e1 + f8_e3;
f9_s1  =  64; f9_e1 = 128; f9_e3 = 128; f9_e = f9_e1 + f9_e3;
conv10 =   2;

PARAMS_DICT = {
	"pConv1W"     : {"dtype": "float32", "shape": ( conv1,      3,  7,  7), "broadcast": (False, False, False, False), "isBias": False},
	"pConv1B"     : {"dtype": "float32", "shape": (     1,  conv1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire2CompW" : {"dtype": "float32", "shape": ( f2_s1,  conv1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire2CompB" : {"dtype": "float32", "shape": (     1,  f2_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire2Exp1W" : {"dtype": "float32", "shape": ( f2_e1,  f2_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire2Exp1B" : {"dtype": "float32", "shape": (     1,  f2_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire2Exp3W" : {"dtype": "float32", "shape": ( f2_e3,  f2_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire2Exp3B" : {"dtype": "float32", "shape": (     1,  f2_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire3CompW" : {"dtype": "float32", "shape": ( f3_s1,  f2_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire3CompB" : {"dtype": "float32", "shape": (     1,  f3_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire3Exp1W" : {"dtype": "float32", "shape": ( f3_e1,  f3_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire3Exp1B" : {"dtype": "float32", "shape": (     1,  f3_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire3Exp3W" : {"dtype": "float32", "shape": ( f3_e3,  f3_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire3Exp3B" : {"dtype": "float32", "shape": (     1,  f3_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire4CompW" : {"dtype": "float32", "shape": ( f4_s1,  f3_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire4CompB" : {"dtype": "float32", "shape": (     1,  f4_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire4Exp1W" : {"dtype": "float32", "shape": ( f4_e1,  f4_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire4Exp1B" : {"dtype": "float32", "shape": (     1,  f4_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire4Exp3W" : {"dtype": "float32", "shape": ( f4_e3,  f4_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire4Exp3B" : {"dtype": "float32", "shape": (     1,  f4_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire5CompW" : {"dtype": "float32", "shape": ( f5_s1,  f4_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire5CompB" : {"dtype": "float32", "shape": (     1,  f5_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire5Exp1W" : {"dtype": "float32", "shape": ( f5_e1,  f5_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire5Exp1B" : {"dtype": "float32", "shape": (     1,  f5_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire5Exp3W" : {"dtype": "float32", "shape": ( f5_e3,  f5_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire5Exp3B" : {"dtype": "float32", "shape": (     1,  f5_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire6CompW" : {"dtype": "float32", "shape": ( f6_s1,  f5_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire6CompB" : {"dtype": "float32", "shape": (     1,  f6_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire6Exp1W" : {"dtype": "float32", "shape": ( f6_e1,  f6_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire6Exp1B" : {"dtype": "float32", "shape": (     1,  f6_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire6Exp3W" : {"dtype": "float32", "shape": ( f6_e3,  f6_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire6Exp3B" : {"dtype": "float32", "shape": (     1,  f6_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire7CompW" : {"dtype": "float32", "shape": ( f7_s1,  f6_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire7CompB" : {"dtype": "float32", "shape": (     1,  f7_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire7Exp1W" : {"dtype": "float32", "shape": ( f7_e1,  f7_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire7Exp1B" : {"dtype": "float32", "shape": (     1,  f7_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire7Exp3W" : {"dtype": "float32", "shape": ( f7_e3,  f7_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire7Exp3B" : {"dtype": "float32", "shape": (     1,  f7_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire8CompW" : {"dtype": "float32", "shape": ( f8_s1,  f7_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire8CompB" : {"dtype": "float32", "shape": (     1,  f8_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire8Exp1W" : {"dtype": "float32", "shape": ( f8_e1,  f8_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire8Exp1B" : {"dtype": "float32", "shape": (     1,  f8_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire8Exp3W" : {"dtype": "float32", "shape": ( f8_e3,  f8_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire8Exp3B" : {"dtype": "float32", "shape": (     1,  f8_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire9CompW" : {"dtype": "float32", "shape": ( f9_s1,  f8_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire9CompB" : {"dtype": "float32", "shape": (     1,  f9_s1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire9Exp1W" : {"dtype": "float32", "shape": ( f9_e1,  f9_s1,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pFire9Exp1B" : {"dtype": "float32", "shape": (     1,  f9_e1,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pFire9Exp3W" : {"dtype": "float32", "shape": ( f9_e3,  f9_s1,  3,  3), "broadcast": (False, False, False, False), "isBias": False},
	"pFire9Exp3B" : {"dtype": "float32", "shape": (     1,  f9_e3,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True },
	"pConv10W"    : {"dtype": "float32", "shape": (conv10,  f9_e ,  1,  1), "broadcast": (False, False, False, False), "isBias": False},
	"pConv10B"    : {"dtype": "float32", "shape": (     1, conv10,  1,  1), "broadcast": ( True, False,  True,  True), "isBias": True }}

#
# KITNN training function entry point
#

KITNN_TRAIN_ENTRY_POINT = "fn"  # FIXME: Change this to the real name!


###############################################################################
# KITNN training session management.

#
# Open a KITNN file.
#

def KFOpen(filePath):
	f = H.File(filePath, "a")           # Create file
	f.require_group("/sessions")        # Ensure a sessions group exists
	return f

#
# Prune inconsistent sessions from a file.
#

def KFPruneInconsistentSessions(f):
	for d in f["/sessions"].keys():
		if f.get("/sessions/"+d+"/meta/consistent", 0)[()] == 0:
			print("Prunning inconsistent session \""+d+"\" ...")
			del f["/sessions/"+d]

#
# Create a consistent session in a file and return it.
#
# Accepts a dictionary of arguments in case this interests the initialization routine.
#
# NOTE: Assumes all sessions in the file are consistent!
#

def KFCreateConsistentSession(f, **kwargs):
	sessions = sorted(f["/sessions"].keys(), key=int)
	
	if len(sessions) == 0:
		n = 1
		
		oldSess = None
		newSess = f.require_group("/sessions/"+str(n))
	else:
		o = sessions[-1]
		n = str(int(o)+1)
		
		oldSess = f.require_group("/sessions/"+str(o))
		newSess = f.require_group("/sessions/"+str(n))
		
	return KSInitSession(newSess, oldSess, **kwargs)

#
# Flush changes made to a file, group or dataset to disk.
#

def KFFlush(h5pyFileOrGroupOrDataset):
	h5pyFileOrGroupOrDataset.file.flush()

#
# Delete unconditionally a path from the file, whether or not it previously
# existed.
#

def KFDeletePaths(h5pyFileOrGroup, pathList):
	if(type(pathList) == str): pathList = [pathList]
	
	for p in pathList:
		if p in h5pyFileOrGroup:
			del h5pyFileOrGroup[p]

#
# Initialize a new session.
#
# If an old, template session is provided, copy the session. It is assumed the old session provided is
# consistent.
#
# If no template session is provided, initialize completely randomly.
#

def KSInitSession(newSess, oldSess, **kwargs):
	# Write the metadata.
	KFDeletePaths(newSess, ["meta/initPRNG",
	                        "meta/src.tar.gz",
	                        "meta/argv",
	                        "meta/unixTimeStarted",
	                        "meta/consistent"])
	
	initPRNG        = np.random.get_state()
	tarGzSrc        = KFSrcTarGz()
	unixTimeStarted = np.full((), time.time(), dtype="float64")
	consistent      = np.full((), 0, dtype="uint64")
	
	KDWritePRNG(newSess.require_group("meta/initPRNG"), data=initPRNG)
	newSess.create_dataset("meta/src.tar.gz",           data=tarGzSrc)
	newSess.create_dataset("meta/argv",                 data=sys.argv, dtype=H5PY_VLEN_STR)
	newSess.create_dataset("meta/unixTimeStarted",      data=unixTimeStarted)
	newSess.create_dataset("meta/consistent",           data=consistent)
	
	# Get or generate the data.
	if oldSess == None:
		#
		# If no sessions exist, initialize snapshot 0/ randomly, then make a
		# copy of it to 1/ and declare 0/ to be the current one using the
		# atomic flag.
		#
		
		KSInitSnapshotRandom(newSess.require_group("snapshot/0"), **kwargs)
		newSess.copy("snapshot/0", "snapshot/1")
		newSess.create_dataset("snapshot/atomic", data=np.full((), 0, dtype="uint64"))
	else:
		#
		# If a session does exist then just copy over the snapshots but not the metadata.
		#
		
		assert(oldSess.get("meta/consistent", 0)[()] == 1)
		oldSess.copy("snapshot", newSess)                 # Copy oldSess/shapshot to newSess/snapshot
	
	# Mark as consistent.
	KFFlush(newSess)
	newSess["meta/consistent"][()] = 1
	KFFlush(newSess)
	
	# Return new session
	return newSess

#
# Write Numpy MT PRNG state to given group.
#

def KDWritePRNG(group, data):
	KFDeletePaths(group, ["name", "keys", "pos", "has_gauss", "cached_gaussian"])
	
	group.create_dataset("name",            data=data[0], dtype=H5PY_VLEN_STR)
	group.create_dataset("keys",            data=data[1])
	group.create_dataset("pos",             data=data[2])
	group.create_dataset("has_gauss",       data=data[3])
	group.create_dataset("cached_gaussian", data=data[4])

#
# Read Numpy MT PRNG state from a given group.
#

def KDReadPRNG(group):
	name            = str     (group["name"           ][()])
	keys            = np.array(group["keys"           ][...])
	pos             = int     (group["pos"            ][()])
	has_gauss       = int     (group["has_gauss"      ][()])
	cached_gaussian = float   (group["cached_gaussian"][()])
	
	return (name, keys, pos, has_gauss, cached_gaussian)

#
# Gzip own source code.
#

def KFSrcTarGz():
	# Get coordinates of files to crush
	kitnnSrcBase = os.path.dirname(os.path.abspath(__file__))
	sourceFiles = ["kitnn.py", "inpainter.cpp"]
	
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

#
# Initialize a snapshot with random weights.
#

def KSInitSnapshotRandom(snap, **kwargs):
	# currKITNN/data folder.
	for (name, desc) in PARAMS_DICT.iteritems():
		dtype     = desc["dtype"]
		shape     = desc["shape"]
		braodcast = desc["broadcast"]
		isBias    = desc["isBias"]
		
		# parameters/ subfolder.
		if isBias:
			value  = np.zeros(shape, dtype)
		else:
			gain   = np.sqrt(2)
			stddev = gain * np.sqrt(2.0 / np.sum(shape[0:2]))
			value  = np.random.normal(scale=stddev, size=shape)
		snap.require_dataset("currKITNN/data/parameters/"+name, shape, dtype, exact=True)[...] = value
		
		# velocities subfolder.
		snap.require_dataset("currKITNN/data/velocities/"+name, shape, dtype, exact=True)[...] = np.zeros_like(value)
	
	# currKITNN/log folder
	KFDeletePaths(snap, ["currKITNN/log/trainLoss",
	                     "currKITNN/log/trainErr",
	                     "currKITNN/log/validErr"])
	snap.require_dataset("currKITNN/log/trainLoss", (0,), "float64", maxshape=(None,))
	snap.require_dataset("currKITNN/log/trainErr",  (0,), "float64", maxshape=(None,))
	snap.require_dataset("currKITNN/log/validErr",  (0,), "float64", maxshape=(None,))
	
	# currKITNN/misc folder.
	snap.require_dataset("currKITNN/misc/mC",        (), H5PY_VLEN_STR, exact=True)[...] = KITNN_TRAIN_ENTRY_POINT
	snap.require_dataset("currKITNN/misc/mE",        (), "uint64", exact=True)[...]      = 0
	snap.require_dataset("currKITNN/misc/mTTI",      (), "uint64", exact=True)[...]      = 0
	snap.require_dataset("currKITNN/misc/mCTI",      (), "uint64", exact=True)[...]      = 0
	snap.require_dataset("currKITNN/misc/mCVI",      (), "uint64", exact=True)[...]      = 0
	snap.require_dataset("currKITNN/misc/mCTErrCnt", (), "uint64", exact=True)[...]      = 0xFFFFFFFFFFFFFFFF
	snap.require_dataset("currKITNN/misc/mCVErrCnt", (), "uint64", exact=True)[...]      = 0xFFFFFFFFFFFFFFFF
	
	# We do PRNG before-last.
	KDWritePRNG(snap.require_group("currKITNN/misc/PRNG"), np.random.get_state())
	
	# Lastly, we copy currKITNN to bestKITNN.
	snap.copy("currKITNN", "bestKITNN")
	
	# Return
	return snap

#
# Init shared variables.
#

def getSVs(source=None):
	if(source != None):
		f = H.File(source, "a")
		
		for (name, param) in f["/param/"].items():
			exec("value_"+name+" = param[...]")
		
		f.close()
	else:
		value_hLrn     = initScalar (0.05)             # Learning Rate
		value_hMom     = initScalar (0.9)              # Momentum
		value_hL1P     = initScalar (0.001)            # L1 penalty
		value_hL2P     = initScalar (0.001)            # L2 penalty
		value_pLaW     = initWeights((  64,   3,3,3))  #   64    3  3   3   convolution weights
		value_pLaB     = initBiases ((   1,  64,1,1))  #   *    64  *   *   biases
		value_pLcW     = initWeights(( 128,  64,3,3))  #  128   64  3   3   convolution weights
		value_pLcB     = initBiases ((   1, 128,1,1))  #   *   128  *   *   biases
		value_pLeW     = initWeights(( 256, 128,3,3))  #  256  128  3   3   convolution weights
		value_pLeB     = initBiases ((   1, 256,1,1))  #   *   256  *   *   biases
		value_pLfW     = initWeights(( 256, 256,3,3))  #  256  256  3   3   convolution weights
		value_pLfB     = initBiases ((   1, 256,1,1))  #   *   256  *   *   biases
		value_pLhW     = initWeights((1024, 256,1,1))  # 1024  256  1   1   convolution weights (Fully-connected)
		value_pLhB     = initBiases ((   1,1024,1,1))  #   *  1024  *   *   biases
		value_pLiW     = initWeights((1024,1024,1,1))  # 1024 1024  1   1   convolution weights (Fully-connected)
		value_pLiB     = initBiases ((   1,1024,1,1))  #   *  1024  *   *   biases
		value_pLiG     = initBiases ((   1,1024,1,1))  #   *  1024  *   *   gammas
		value_pLjW     = initWeights((  10,1024,1,1))  #   10 1024  1   1   convolution weights (Fully-connected)
		value_pLjB     = initBiases ((   1,  10,1,1))  #   *    10  *   *   biases
	
	hLrn           = T.shared(value_hLrn,  name="hLrn")
	hMom           = T.shared(value_hMom,  name="hMom")
	hL1P           = T.shared(value_hL1P,  name="hL1P")
	hL2P           = T.shared(value_hL2P,  name="hL2P")
	pLaW           = T.shared(value_pLaW,  name="pLaW")
	pLaB           = T.shared(value_pLaB,  name="pLaB",  broadcastable=(True, False, True, True))
	pLcW           = T.shared(value_pLcW,  name="pLcW")
	pLcB           = T.shared(value_pLcB,  name="pLcB",  broadcastable=(True, False, True, True))
	pLeW           = T.shared(value_pLeW,  name="pLeW")
	pLeB           = T.shared(value_pLeB,  name="pLeB",  broadcastable=(True, False, True, True))
	pLfW           = T.shared(value_pLfW,  name="pLfW")
	pLfB           = T.shared(value_pLfB,  name="pLfB",  broadcastable=(True, False, True, True))
	pLhW           = T.shared(value_pLhW,  name="pLhW")
	pLhB           = T.shared(value_pLhB,  name="pLhB",  broadcastable=(True, False, True, True))
	pLiW           = T.shared(value_pLiW,  name="pLiW")
	pLiB           = T.shared(value_pLiB,  name="pLiB",  broadcastable=(True, False, True, True))
	pLiG           = T.shared(value_pLiG,  name="pLiG",  broadcastable=(True, False, True, True))
	pLjW           = T.shared(value_pLjW,  name="pLjW")
	pLjB           = T.shared(value_pLjB,  name="pLjB",  broadcastable=(True, False, True, True))
	
	SVs = {}
	for sv in [hLrn, hMom, hL1P, hL2P, pLaW, pLaB, pLcW, pLcB, pLeW, pLeB,
	           pLfW, pLfB, pLhW, pLhB, pLiW, pLiB, pLiG, pLjW, pLjB]:
		SVs[sv.name] = sv
	
	return SVs


#
# Dump SVs to HDF5
#

def dumpSVs(source, SV):
	f = H.File(source, "a")
	
	for (name, param) in SV.iteritems():
		value = param.get_value()
		f.require_dataset("/param/"+name,
		                  value.shape,
		                  value.dtype,
		                  exact=True)[...] = value
	
	f.close()


#
# Construct Theano functions.
#
# Need two:
#  1. For training, which has inputs ix and iy, returns the loss, and updates
#     the parameters.
#  2. For classification, which has inputs ix and returns the softmax out.
#

def constructTheanoFuncs(SV):
	#
	# Classification function construction.
	#
	
	# Input is ix.
	ix = TT.tensor4("x") # (Batch=hB, #Channels=3, Height=192, Width=192)
	
	
	##########################################################
	# The math.                                              #
	##########################################################
	
	######################  Input layer
	vIn           = ix
	
	######################  conv1
	vConv1Act     = TTNC.conv2d (vIn, pConv1W, None, None, "half", (2,2)) + pConv1B
	vConv1        = TTN .relu   (vConv1Act)
	
	######################  maxpool1
	vMaxpool1     = TTSP.pool_2d(vLconv1, (3,3), False, (2,2), "max")
	
	######################  fire2
	vFire2CompAct = TTNC.conv2d (vMaxpool1,  pFire2CompW, None, None, "half", (1,1)) + pFire2CompB
	vFire2Comp    = TTN .relu   (vFire2CompAct)
	vFire2Exp1Act = TTNC.conv2d (vFire2Comp, pFire2Exp1W, None, None, "half", (1,1)) + pFire2Exp1B
	vFire2Exp1    = TTN .relu   (vFire2Exp1Act)
	vFire2Exp3Act = TTNC.conv2d (vFire2Comp, pFire2Exp3W, None, None, "half", (1,1)) + pFire2Exp3B
	vFire2Exp3    = TTN .relu   (vFire2Exp3Act)
	vFire2        = TT  .stack  ([vFire2Exp1, vFire2Exp3], axis=1);
	
	######################  fire3
	vFire3CompAct = TTNC.conv2d (vFire2,     pFire3CompW, None, None, "half", (1,1)) + pFire3CompB
	vFire3Comp    = TTN .relu   (vFire3CompAct)
	vFire3Exp1Act = TTNC.conv2d (vFire3Comp, pFire3Exp1W, None, None, "half", (1,1)) + pFire3Exp1B
	vFire3Exp1    = TTN .relu   (vFire3Exp1Act)
	vFire3Exp3Act = TTNC.conv2d (vFire3Comp, pFire3Exp3W, None, None, "half", (1,1)) + pFire3Exp3B
	vFire3Exp3    = TTN .relu   (vFire3Exp3Act)
	vFire3        = TT  .stack  ([vFire3Exp1, vFire3Exp3], axis=1);
	
	######################  fire4
	vFire4CompAct = TTNC.conv2d (vFire3,     pFire4CompW, None, None, "half", (1,1)) + pFire4CompB
	vFire4Comp    = TTN .relu   (vFire4CompAct)
	vFire4Exp1Act = TTNC.conv2d (vFire4Comp, pFire4Exp1W, None, None, "half", (1,1)) + pFire4Exp1B
	vFire4Exp1    = TTN .relu   (vFire4Exp1Act)
	vFire4Exp3Act = TTNC.conv2d (vFire4Comp, pFire4Exp3W, None, None, "half", (1,1)) + pFire4Exp3B
	vFire4Exp3    = TTN .relu   (vFire4Exp3Act)
	vFire4        = TT  .stack  ([vFire4Exp1, vFire4Exp3], axis=1);
	
	######################  maxpool4
	vMaxpool4     = TTSP.pool_2d(vFire4, (3,3), False, (2,2), "max")
	
	######################  fire5
	vFire5CompAct = TTNC.conv2d (vMaxpool4,  pFire5CompW, None, None, "half", (1,1)) + pFire5CompB
	vFire5Comp    = TTN .relu   (vFire5CompAct)
	vFire5Exp1Act = TTNC.conv2d (vFire5Comp, pFire5Exp1W, None, None, "half", (1,1)) + pFire5Exp1B
	vFire5Exp1    = TTN .relu   (vFire5Exp1Act)
	vFire5Exp3Act = TTNC.conv2d (vFire5Comp, pFire5Exp3W, None, None, "half", (1,1)) + pFire5Exp3B
	vFire5Exp3    = TTN .relu   (vFire5Exp3Act)
	vFire5        = TT  .stack  ([vFire5Exp1, vFire5Exp3], axis=1);
	
	######################  fire6
	vFire6CompAct = TTNC.conv2d (vFire5,     pFire6CompW, None, None, "half", (1,1)) + pFire6CompB
	vFire6Comp    = TTN .relu   (vFire6CompAct)
	vFire6Exp1Act = TTNC.conv2d (vFire6Comp, pFire6Exp1W, None, None, "half", (1,1)) + pFire6Exp1B
	vFire6Exp1    = TTN .relu   (vFire6Exp1Act)
	vFire6Exp3Act = TTNC.conv2d (vFire6Comp, pFire6Exp3W, None, None, "half", (1,1)) + pFire6Exp3B
	vFire6Exp3    = TTN .relu   (vFire6Exp3Act)
	vFire6        = TT  .stack  ([vFire6Exp1, vFire6Exp3], axis=1);
	
	######################  fire7
	vFire7CompAct = TTNC.conv2d (vFire6,     pFire7CompW, None, None, "half", (1,1)) + pFire7CompB
	vFire7Comp    = TTN .relu   (vFire7CompAct)
	vFire7Exp1Act = TTNC.conv2d (vFire7Comp, pFire7Exp1W, None, None, "half", (1,1)) + pFire7Exp1B
	vFire7Exp1    = TTN .relu   (vFire7Exp1Act)
	vFire7Exp3Act = TTNC.conv2d (vFire7Comp, pFire7Exp3W, None, None, "half", (1,1)) + pFire7Exp3B
	vFire7Exp3    = TTN .relu   (vFire7Exp3Act)
	vFire7        = TT  .stack  ([vFire7Exp1, vFire7Exp3], axis=1);
	
	######################  fire8
	vFire8CompAct = TTNC.conv2d (vFire7,     pFire8CompW, None, None, "half", (1,1)) + pFire8CompB
	vFire8Comp    = TTN .relu   (vFire8CompAct)
	vFire8Exp1Act = TTNC.conv2d (vFire8Comp, pFire8Exp1W, None, None, "half", (1,1)) + pFire8Exp1B
	vFire8Exp1    = TTN .relu   (vFire8Exp1Act)
	vFire8Exp3Act = TTNC.conv2d (vFire8Comp, pFire8Exp3W, None, None, "half", (1,1)) + pFire8Exp3B
	vFire8Exp3    = TTN .relu   (vFire8Exp3Act)
	vFire8        = TT  .stack  ([vFire8Exp1, vFire8Exp3], axis=1);
	
	######################  maxpool8
	vMaxpool8     = TTSP.pool_2d(vFire8, (3,3), False, (2,2), "max")
	
	######################  fire9
	vFire9CompAct = TTNC.conv2d (vMaxpool8,  pFire9CompW, None, None, "half", (1,1)) + pFire9CompB
	vFire9Comp    = TTN .relu   (vFire9CompAct)
	vFire9Exp1Act = TTNC.conv2d (vFire9Comp, pFire9Exp1W, None, None, "half", (1,1)) + pFire9Exp1B
	vFire9Exp1    = TTN .relu   (vFire9Exp1Act)
	vFire9Exp3Act = TTNC.conv2d (vFire9Comp, pFire9Exp3W, None, None, "half", (1,1)) + pFire9Exp3B
	vFire9Exp3    = TTN .relu   (vFire9Exp3Act)
	vFire9        = TT  .stack  ([vFire9Exp1, vFire9Exp3], axis=1);
	
	######################  conv10
	vConv10Act    = TTNC.conv2d (vFire9, pConv10W, None, None, "half", (1,1)) + pConv10B
	vConv10       = TTN .relu   (vConv10Act)
	
	######################  avgpool10
	vAvgpool10    = TTSP.pool_2d(vConv10, (12,12), True, (1,1), "average_exc_pad")
	
	######################  Softmax
	vSMi          = vAvgpool10
	vSMu          = TT.exp(vSMi - TT.max(vSMi, axis=1, keepdims=1))
	vSM           =        vSMu / TT.sum(vSMu, axis=1, keepdims=1)
	
	######################  Output layer
	oy           = vSM
	
	# Function creation
	classf       = T.function(inputs=[ix], outputs=oy, name="classification-function")
	
	
	
	
	#
	# Training function construction.
	#
	
	# Inputs also include iy.
	iy = TT.tensor4("y") # (Batch=hB, #Classes=10, Height=1, Width=1)
	
	
	######################  Regularization
	L1decay      = TT.sum(TT.abs_(SV["pLaW"])) + TT.sum(TT.abs_(SV["pLcW"])) + TT.sum(TT.abs_(SV["pLeW"])) + \
	               TT.sum(TT.abs_(SV["pLfW"])) + TT.sum(TT.abs_(SV["pLhW"])) + TT.sum(TT.abs_(SV["pLiW"])) + \
	               TT.sum(TT.abs_(SV["pLjW"]))
	L2decay      = TT.sum(SV["pLaW"]*SV["pLaW"]) + TT.sum(SV["pLcW"]*SV["pLcW"]) + TT.sum(SV["pLeW"]*SV["pLeW"]) + \
	               TT.sum(SV["pLfW"]*SV["pLfW"]) + TT.sum(SV["pLhW"]*SV["pLhW"]) + TT.sum(SV["pLiW"]*SV["pLiW"]) + \
	               TT.sum(SV["pLjW"]*SV["pLjW"])
	
	
	######################  Cross-Entropy Loss
	oceloss      = TT.sum(TT.mean(-iy*TT.log(oy), axis=0))           # Average across batch, sum over space.
	oloss        = oceloss # + L2decay*SV["hL2P"] + L1decay*SV["hL1P"]
	
	
	######################  Update rules & Gradients
	updates = []
	for (name, param) in SV.iteritems():
		if(name.startswith("p")):
			#
			# It's a parameter, so we do SGD with momentum on it.
			#
			# To do this we get the gradient and construct a velocity shared variable.
			#
			
			pgrad = T.grad(oloss, param)
			pvel  = T.shared(np.zeros(param.get_value().shape, dtype=param.dtype),
			                 name          = param.name+"vel",
			                 broadcastable = param.broadcastable)
			
			# Momentum rule:
			newpvel  = SV["hMom"]*pvel + (1.0-SV["hMom"])*pgrad
			newparam = param - SV["hLrn"]*newpvel
			
			#Updates
			updates.append((pvel,
			                newpvel))
			updates.append((param,
			                newparam))
	
	
	# Function creation
	lossf  = T.function(inputs=[ix, iy], outputs=oloss, updates=updates, name="loss-function")
	
	
	#TP.pydotprint(lossf, "graph.png", format="png", with_ids=False, compact=True)
	return (classf, lossf)





###############################################################################
# KITNN Trainer class code.
#

class KITNNTrainer(object):
	#
	# Construct a trainer object from arguments.
	#
	
	def __init__(self, argv):
		self.kEpoch = 0
		
		#
		# IMPORTANT:
		#
		# The below is a critical part of the continuation-passing style used
		# in train(). It's what allows finegrained resumable training.
		#
		
		if False:
			self.SETCC(None) #FIXME: Whatever we loaded from the save file!
		else:
			self.SETCC(self.ccStartEpoch);
	
	#
	# IMPLEMENTATION OF CONTINUATION PASSING STYLE.
	#
	# Three functions implement our mini-CPS training environment.
	#
	# - Have continuation?
	#
	#     We have a continuation if self.cc is a tuple of length 4. If so:
	#       - Its first element is a callable continuation function.
	#       - Its second element is a tuple of arguments to be passed to the
	#         continuation
	#       - Its third element is a return value.
	#       - Its fourth element is a continue Boolean flag that indicates
	#         whether we're continuing, or returning from the trampoline.
	#
	# - Invoke continuation.
	# - Set/Return continuation.
	#
	#     - If the "continue" flag is True, we are "returning" only the
	#       continuation which the trampoline is expected to call.
	#     - If the "continue" flag is False, we are truly returning a value an
	#       breaking out of the trampoline.
	#
	# - Set continuation arguments.
	#
	#    Intended to be used by resuming code. Sets the current
	#    continuation's arguments and sets the continue flag to True.
	#    After a suspension of training, this allows a subsequent RUNCC() to
	#    resume training.
	#
	# - Run continuation trampoline.
	#
	#     Runs the trampoline loop that implements our CPS style.
	#
	
	def HAVECC(self):
		return len     (self.cc   ) ==     4 and \
		       callable(self.cc[0]) ==  True and \
		       type    (self.cc[1]) == tuple and \
		       type    (self.cc[3]) ==  bool and \
			   self.cc[3]           ==  True
	
	def INVKCC(self):
		if self.HAVECC():
			self.cc[0](*self.cc[1])
		return self.cc[2]
	
	def SETCC(self, fun, args=()):
		assert callable(fun)
		assert type(args)==tuple
		self.cc = (fun, args, None, True)
		return self.cc
	
	def RETCC(self, ret=None, fun=None, args=()):
		if fun==None:
			self.cc = (None, (), ret, False)
		else:
			assert callable(fun)
			assert type(args)==tuple
			
			self.cc = (fun, args, ret, False)
		return self.cc
	
	def SETCCARGS(self, args):
		assert type(args)==tuple
		self.cc[2] = args
		self.cc[3] = True
	
	def RUNCC(self):
		ret = None
		try:
			while self.HAVECC():
				ret = self.INVKCC()
		except KeyboardInterrupt as kbdie:
			print("Stopped.")
		finally:
			return ret
	
	#
	# Train a KITNN.
	#
	
	def train(self):
		self.RUNCC()
	
	#
	# Start an epoch
	#
	
	def ccStartEpoch(self):
		self.ccDoTrainPass()
		self.ccDoBNPass()
		self.ccDoValidPass()
		self.kEpoch += 1
		
		print self.kEpoch
		
		if self.kEpoch<10:
			return self.SETCC(self.ccStartEpoch)
		else:
			return self.RETCC(None, self.ccStartEpoch)
	
	
	#
	# Do one pass over the training set.
	#
	
	def ccDoTrainPass(self):
		pass
	
	
	#
	# Learn Batch Normalization constants over training set.
	#
	
	def ccDoBNPass(self):
		pass
	
	
	#
	# Do one pass over the validation set.
	#
	
	def ccDoValidPass(self):
		pass
	
	
	#
	# Print status
	#
	
	def setAndPrintState(self, newState=None, newLine=False, doPrint=True):
		# Set current state to new state
		if(newState != None):
			self.state = newState
		
		# Print if told to
		if(doPrint):
			# Switch on current state.
			if  (self.state == "a"):
				sys.stdout.write()
			elif(self.state == "b"):
				sys.stdout.write()
			elif(self.state == "c"):
				sys.stdout.write()
			elif(self.state == "d"):
				sys.stdout.write()
			elif(self.state == "e"):
				sys.stdout.write()
			elif(self.state == "f"):
				sys.stdout.write()
			
			# Final newline, if wanted
			if newLine:
				sys.stdout.write("\n")
			
			# Flush
			sys.stdout.flush()





###############################################################################
# KITNN Class code.
#

class KITNN(object):
	#
	# Construct a KITNN object.
	#
	
	def __init__(self, sess):
		pass
	
	#
	# Load oneself from an HDF5 group.
	#
	# This operation must be idempotent.
	#
	
	def load(self, sess):
		pass
	
	#
	# Save oneself to an HDF5 group.
	#
	# This operation must be idempotent.
	#
	
	def save(self, sess):
		pass
	
	#
	# Classify image(s).
	#
	# Accepts a (B,3,H,W)-shaped tensor of B images, and returns a (B,C)-shaped
	# tensor of C class probabilities for each of the B images.
	#
	
	def classify(self, imgs):
		pass
	
	#
	# Train update.
	#
	# Performs forwardprop, backprop and update by invoking the training
	# function.
	#
	# Returns the training function's return values.
	#
	
	def update(self, **kwargs):
		pass






###############################################################################
# Implementations of the script's "verbs".
#

#
# Print help/usage information
#

def verb_help(argv=None):
	print(
"""
Usage of KITNN.

The KITNN script is invoked using a verb that denotes the general action to be
taken, plus optional arguments. The following verbs are defined:

    \033[1mclassify\033[0m:

    \033[1mhelp\033[0m:
        This help message.

    \033[1mtrain\033[0m:

"""[1:-1] #This hack deletes the newlines before and after the triple quotes.
	)
	
	pdb.set_trace()



#
# Classify the argument images as cat or dog.
#

def verb_classify(argv):
	print argv



#
# Extract code and arguments from the session file. Print the args to stderr
# and the .tar.gz to stdout.
#

def verb_extractCode(argv):
	print argv



#
# Train KITNN.
#

def verb_train(argv=None):
	kitnntrainer = KITNNTrainer(argv)
	if(kitnntrainer == None):
		exit(1)
	
	kitnntrainer.train()
	
	#pdb.set_trace()




#
# Screw around.
#

def verb_screw(argv=None):
	f = KFOpen(argv[2])
	KFPruneInconsistentSessions(f)
	s = KFCreateConsistentSession(f)


#
# Screw around.
#

def verb_interactive(argv=None):
	pdb.set_trace()


#
# Dump parameter dict
#

def verb_dumpparamdict(argv):
	totalParams = 0
	
	for (k,v) in sorted(PARAMS_DICT.iteritems()):
		vShape        = v["shape"]
		vNumParams    = np.prod(vShape)
		totalParams  += vNumParams
		print("{:20s}: {:10d}".format(k, vNumParams))
	print("{:20s}: {:10d}".format("TOTAL", totalParams))


#
# Dump source code of session
#

def verb_dumpsrcs(argv):
	f  = H.File(argv[2], "r")
	gz = bytearray(f["/sessions/"+argv[3]+"/meta/src.tar.gz"])
	sys.stdout.write(gz)
	f.close()

#
# Dump arguments of session
#

def verb_dumpargs(argv):
	f  = H.File(argv[2], "r")
	print list(f["/sessions/"+argv[3]+"/meta/argv"][...])
	f.close()



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



# pool_2d(input, ds, ignore_border=None, st=None, padding=(0, 0), mode='max')
#     Downscale the input by a specified factor
#
#     Takes as input a N-D tensor, where N >= 2. It downscales the input image by
#     the specified factor, by keeping only the maximum value of non-overlapping
#     patches of size (ds[0],ds[1])
#
#     Parameters
#     ----------
#     input : N-D theano tensor of input images
#         Input images. Max pooling will be done over the 2 last dimensions.
#     ds : tuple of length 2
#         Factor by which to downscale (vertical ds, horizontal ds).
#         (2,2) will halve the image in each dimension.
#     ignore_border : bool (default None, will print a warning and set to False)
#         When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
#         (3,3) otherwise.
#     st : tuple of two ints
#         Stride size, which is the number of shifts over rows/cols to get the
#         next pool region. If st is None, it is considered equal to ds
#         (no overlap on pooling regions).
#     padding : tuple of two ints
#         (pad_h, pad_w), pad zeros to extend beyond four borders of the
#         images, pad_h is the size of the top and bottom margins, and
#         pad_w is the size of the left and right margins.
#     mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
#         Operation executed on each window. `max` and `sum` always exclude
#         the padding in the computation. `average` gives you the choice to
#         include or exclude it.



# conv2d(input, filters, input_shape=None, filter_shape=None, border_mode='valid', subsample=(1, 1), filter_flip=True, image_shape=None, **kwargs)
#     This function will build the symbolic graph for convolving a mini-batch of a
#     stack of 2D inputs with a set of 2D filters. The implementation is modelled
#     after Convolutional Neural Networks (CNN).
#
#
#     Parameters
#     ----------
#     input: symbolic 4D tensor
#         Mini-batch of feature map stacks, of shape
#         (batch size, input channels, input rows, input columns).
#         See the optional parameter ``input_shape``.
#
#     filters: symbolic 4D tensor
#         Set of filters used in CNN layer of shape
#         (output channels, input channels, filter rows, filter columns).
#         See the optional parameter ``filter_shape``.
#
#     input_shape: None, tuple/list of len 4 of int or Constant variable
#         The shape of the input parameter.
#         Optional, possibly used to choose an optimal implementation.
#         You can give ``None`` for any element of the list to specify that this
#         element is not known at compile time.
#
#     filter_shape: None, tuple/list of len 4 of int or Constant variable
#         The shape of the filters parameter.
#         Optional, possibly used to choose an optimal implementation.
#         You can give ``None`` for any element of the list to specify that this
#         element is not known at compile time.
#
#     border_mode: str, int or tuple of two int
#         Either of the following:
#
#         ``'valid'``: apply filter wherever it completely overlaps with the
#             input. Generates output of shape: input shape - filter shape + 1
#         ``'full'``: apply filter wherever it partly overlaps with the input.
#             Generates output of shape: input shape + filter shape - 1
#         ``'half'``: pad input with a symmetric border of ``filter rows // 2``
#             rows and ``filter columns // 2`` columns, then perform a valid
#             convolution. For filters with an odd number of rows and columns, this
#             leads to the output shape being equal to the input shape.
#         ``int``: pad input with a symmetric border of zeros of the given
#             width, then perform a valid convolution.
#         ``(int1, int2)``: pad input with a symmetric border of ``int1`` rows
#             and ``int2`` columns, then perform a valid convolution.
#
#     subsample: tuple of len 2
#         Factor by which to subsample the output.
#         Also called strides elsewhere.
#
#     filter_flip: bool
#         If ``True``, will flip the filter rows and columns
#         before sliding them over the input. This operation is normally referred
#         to as a convolution, and this is the default. If ``False``, the filters
#         are not flipped and the operation is referred to as a cross-correlation.
#
#     image_shape: None, tuple/list of len 4 of int or Constant variable
#         Deprecated alias for input_shape.
#
#     kwargs: Any other keyword arguments are accepted for backwards
#             compatibility, but will be ignored.
#
#     Returns
#     -------
#     Symbolic 4D tensor
#         Set of feature maps generated by convolutional layer. Tensor is
#         of shape (batch size, output channels, output rows, output columns)
#
#     Notes
#     -----
#         If CuDNN is available, it will be used on the
#         GPU. Otherwise, it is the *CorrMM* convolution that will be used
#         "caffe style convolution".
#
#         This is only supported in Theano 0.8 or the development
#         version until it is released.
