#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module Docstring
"""
KITNN: The KITten Neural Network, by Olexa Bilaniuk.

A corny name, as is customary for computer scientists, given to my CNN that
attempts to solve the problem of correctly classifying Cats & Dogs for the
Kaggle challenge. This work is done as part of the course project for the
IFT6266 class given in Winter 2016.

<Describe CNN layer structure here>
"""

#
# Imports
#

import ast
import cPickle as pkl
import cv2
import getopt
import gzip
import h5py
import inspect
import math
import numpy as np
import os
import pdb
import sys
import theano                          as T
import theano.tensor                   as TT
import theano.tensor.nnet              as TTN
import theano.tensor.nnet.conv         as TTNC
import theano.tensor.signal.downsample as TTSD
from   theano import config            as TC
import theano.printing                 as TP
import time




###############################################################################
# KITNN code.
#

# Global constants (hopefully).
H5PY_VLEN_STR = h5py.special_dtype(vlen=str)

#
# Get the source code of this module (the contents of this very file) and the
# arguments that were passed to it.
#

def getArgsAndSrcs():
	args = sys.argv
	srcs = inspect.getsource(sys.modules[__name__])
	
	return (args, srcs)

#
# Save arguments and source code into given HDF5 group.
#

def saveArgsAndSrcs(group, args, src):
	# Store arguments.
	group.create_dataset("args", dtype=H5PY_VLEN_STR, data=args)
	
	# Store source code.
	group.create_dataset("src",  dtype=H5PY_VLEN_STR, data=src)




#
# Default parameter store
#

DEFAULT_FILEPATH = "params.hdf5"


#
# Get shared-variable source.
#
# For now, must be none.
#

def getSource(filepath=DEFAULT_FILEPATH):
	if(os.path.exists(filepath)):
		return filepath
	else:
		return None


#
# Initialize a weights vector using uniform distribution, appropriately ranged.
#

def initWeights(size):
    n = np.prod(size)
    c = np.sqrt(2.0/n)
    return np.random.normal(0, c, size).astype(TC.floatX)

#
# Initialize a bias vector using ones.
#

def initBiases(size):
    return np.zeros(size, dtype=TC.floatX)

#
# Initialize a float scalar.
#

def initScalar(s):
    sv = np.zeros((), dtype=TC.floatX)
    sv[()] = s
    return sv

#
# Init shared variables.
#

def getSVs(source=None):
	if(source != None):
		f = h5py.File(source, "a")
		
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
	f = h5py.File(source, "a")
	
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
	ix = TT.tensor4("x") # (Batch=hB, #Channels=3, Height=32, Width=32)
	
	
	##########################################################
	# The math.                                              #
	##########################################################
	
	
	######################  Input layer
	vLin         = ix
	
	######################  Layer a
	vLaAct       = TTNC.conv2d(input=vLin, filters=SV["pLaW"]) + SV["pLaB"]
	vLa          = TTN.relu(vLaAct)
	
	######################  Layer b
	vLb          = TTSP.pool_2d(vLa, ds=(2,2), ignore_border=True, st=(2,2), mode="max")
	
	######################  Layer c
	vLcAct       = TTNC.conv2d(input=vLb, filters=SV["pLcW"]) + SV["pLcB"]
	vLc          = TTN.relu(vLcAct)
	
	######################  Layer d
	vLd          = TTSP.pool_2d(vLc, ds=(2,2), ignore_border=True, st=(2,2), mode="max")
	
	######################  Layer e
	vLeAct       = TTNC.conv2d(input=vLd, filters=SV["pLeW"]) + SV["pLeB"]
	vLe          = TTN.relu(vLeAct)
	
	######################  Layer f
	vLfAct       = TTNC.conv2d(input=vLe, filters=SV["pLfW"]) + SV["pLfB"]
	vLf          = TTN.relu(vLfAct)
	
	######################  Layer g
	vLg          = TTSP.pool_2d(vLf, ds=(2,2), ignore_border=True, st=(2,2), mode="max")
	
	######################  Layer h
	vLhAct       = TTNC.conv2d(input=vLg, filters=SV["pLhW"]) + SV["pLhB"]
	vLh          = TTN.relu(vLhAct)
	
	######################  Layer i
	#vLiAct       = TTNC.conv2d(input=vLh, filters=SV["pLiW"]) + SV["pLiB"]
	#vLi          = TTN.relu(vLiAct)
	vLiAct       = TTNC.conv2d(input=vLh, filters=SV["pLiW"])
	vLiBn        = TTNB.batch_normalization(vLiAct, SV["pLiG"], SV["pLiB"], TT.mean(vLiAct, axis=0, keepdims=1), TT.std(vLiAct, axis=0, keepdims=1))
	vLi          = TTN.relu(vLiBn)
	
	######################  Layer j
	vLj          = TTNC.conv2d(input=vLi, filters=SV["pLjW"]) + SV["pLjB"]
	
	######################  Softmax
	vLkunnorm    = TT.exp(vLj - vLj.max(axis=1, keepdims=1))
	vLk          = vLkunnorm / vLkunnorm.sum(axis=1, keepdims=1)
	
	######################  Output layer
	oy           = vLk
	
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
# Implementations of the script's "verbs".
#

#
# Annotate dataset with bounding boxes.
#

def verb_annotatebbds(argv):
	# Check arguments
	if(len(argv) != 4):
		return verb_help()
	
	# The first argument is the annotations file, the second is the dataset
	# directory.
	annfile = argv[2]
	dataset = argv[3]
	
	#Sanity check their existence or accessibility
	if(os.path.exists(annfile)):
		if(not os.path.isfile(annfile)):
			print("{:s} is not a file!".format(annfile))
			return
		
		if(not os.access(annfile, os.W_OK)):
			print("{:s} is not writable!".format(annfile))
			return
	else:
		catlist = [[] for x in xrange(12500)]
		doglist = [[] for x in xrange(12500)]
		f = open(annfile, "w+")
		pkl.dump((catlist, doglist), f, -1)
		f.close()
	
	f = open(annfile, "r+")
	pkl.dump((catlist, doglist), f, -1)
	f.close()
	
	if(not os.path.isdir (dataset)):
		print("{:s} is not a directory!".format(dataset))
		return
	
	# Variables we'll manipulate every iteration
	classes = ["cat", "dog"]
	c      = 0                    # Species
	i      = 1                    # Image #
	j      = -1                   # Bounding Box #
	num    = ""                   # Entered number.
	
	#
	# Loop until we decide to quit.
	#
	
	while True:
		#
		# A loop iteration has three steps.
		#
		# 1) Draw the situation at (c,i). The current bounding box, if any, is
		#    drawn with linesize 2; The rest (if any) with linesize 1.
		# 2) Display the situation and wait for key input.
		# 3) Handle the keyboard input.
		#
		
		if(c <     0): c =     0
		if(c >     1): c =     1
		if(i <     1): c =     1
		if(i > 12500): c = 12500
		
		imgpathname = os.path.join(dataset, "{:s}.{:d}.jpg".format(classes[c], i))
		img         = cv2.imread(imgpathname)
		cv2.imshow("Image", img)
		key         = cv2.waitKey()
		
		print(chr(key))
		
		break

#
# Classify the argument images as cat or dog.
#

def verb_classify(argv):
	print argv

#
# Print help/usage information
#

def verb_help(argv=None):
	print(
"""
Usage of KITNN.

The KITNN script is invoked using a verb that denotes the general action to be
taken, plus optional arguments. The following verbs are defined:

    \033[1mannotatebbds\033[0m:

    \033[1mclassify\033[0m:

    \033[1mhelp\033[0m:
        This help message.

    \033[1mtrain\033[0m:

"""[1:-1] #This hack deletes the newlines before and after the triple quotes.
	)

#
# Train KITNN.
#

def verb_train(args=None):
	# Theano code.
	source          = getSource()
	SV              = getSVs(source)
	(classf, lossf) = constructTheanoFuncs(SV)
	
	
	
	# Load data
	(ix, iy) = unpickleCIFAR("cifar-10-batches-py")
	
	# Dump image and its flip
	img = ix[2]
	img = img.reshape((3,32,32))
	img = np.einsum("cyx->yxc", img)[:,:,::-1]   # Transpose, then reverse channel order, since OpenCV uses BGR
	cv2.imwrite("ImageOriginal.png", img)
	cv2.imwrite("ImageFlipped.png",  img[:,::-1,:])
	
	# Split dataset
	T = 45000
	V =  5000
	train_ix = ix[:T]
	train_iy = iy[:T]
	valid_ix = ix[T:T+V]
	valid_iy = iy[T:T+V]
	test_ix  = ix[T+V:]
	test_iy  = iy[T+V:]
	
	# Run Main Loop
	iter             = 0
	batch_size_train = 32
	batch_size_valid = 1000
	
	#
	# 20 Epochs
	#
	
	for e in xrange(20):
		# Timing start.
		ts = time.time()
		
		#
		# Train over whole batch
		#
		
		B=0
		while((B+1)*batch_size_train <= T):
			# Compute selection
			Bs = (B  )*batch_size_train
			Be = (B+1)*batch_size_train
			
			# Extract and convert as needed
			sel_ix = train_ix[Bs:Be]
			sel_iy = train_iy[Bs:Be]
			
			# Flip horizontally as needed
			for flipidx in xrange(batch_size_train):
				if(np.random.randint(2) == 1):
					sel_ix[flipidx] = sel_ix[flipidx,:,:,::-1]
			
			# Run trainer
			loss = lossf(sel_ix, sel_iy)[()]
			
			# Print iteration and loss function
			print("Epoch {:3d}      Iter {:7d}     Loss {:12.6f}".format(e, iter, loss))
			
			# Increment counters.
			iter += 1
			B    += 1
		
		#
		# Evaluate train and validation error
		#
		
		train_err = 0
		
		print("                                 **** VALIDATING...           ****");
		B=0
		valid_err  = 0
		valid_loss = 0
		while((B+1)*batch_size_valid <= len(valid_ix)):
			# Compute selection
			Bs = (B  )*batch_size_valid
			Be = (B+1)*batch_size_valid
			
			# Extract and convert as needed
			sel_ix = valid_ix[Bs:Be]
			sel_iy = valid_iy[Bs:Be]
			
			# Run classifier
			out_iy = classf(sel_ix)
			
			# Argmax
			valid_loss += -np.sum(sel_iy * np.log(out_iy))
			out_iy = np.argmax(out_iy[:,:,0,0], axis=1)
			sel_iy = np.argmax(sel_iy[:,:,0,0], axis=1)
			
			#Accumulate errors
			valid_err += np.sum(out_iy != sel_iy)
			
			#Increment Counter
			B    += 1
		
		# Compute average
		valid_err  = float(valid_err)  / len(valid_ix)
		valid_loss = float(valid_loss) / len(valid_ix)
		
		# Print
		print("                                 **** VALID ERR: {:12.6f}    LOSS: {:12.6f} ****".format(valid_err*100.0, valid_loss));
		
		
		
		print("                                 **** TESTING...              ****");
		B=0
		test_err  = 0
		test_loss = 0
		while((B+1)*batch_size_valid <= len(test_ix)):
			# Compute selection
			Bs = (B  )*batch_size_valid
			Be = (B+1)*batch_size_valid
			
			# Extract and convert as needed
			sel_ix = test_ix[Bs:Be]
			sel_iy = test_iy[Bs:Be]
			
			# Run classifier
			out_iy = classf(sel_ix)
			
			# Argmax
			test_loss += -np.sum(sel_iy * np.log(out_iy))
			out_iy = np.argmax(out_iy[:,:,0,0], axis=1)
			sel_iy = np.argmax(sel_iy[:,:,0,0], axis=1)
			
			#Accumulate errors
			test_err += np.sum(out_iy != sel_iy)
			
			#Increment Counter
			B    += 1
		
		# Compute average
		test_err  = float(test_err)  / len(test_ix)
		test_loss = float(test_loss) / len(test_ix)
		
		# Print
		print("                                 **** TEST  ERR: {:12.6f}    LOSS: {:12.6f} ****".format(test_err*100.0, test_loss));
		
		
		
		print("                                 **** TRAIN ERR...             ****");
		B=0
		train_err  = 0
		train_loss = 0
		while((B+1)*batch_size_valid <= len(train_ix)):
			# Compute selection
			Bs = (B  )*batch_size_valid
			Be = (B+1)*batch_size_valid
			
			# Extract and convert as needed
			sel_ix = train_ix[Bs:Be]
			sel_iy = train_iy[Bs:Be]
			
			# Run classifier
			out_iy = classf(sel_ix)
			
			# Argmax
			train_loss += -np.sum(sel_iy * np.log(out_iy))
			out_iy = np.argmax(out_iy[:,:,0,0], axis=1)
			sel_iy = np.argmax(sel_iy[:,:,0,0], axis=1)
			
			#Accumulate errors
			train_err += np.sum(out_iy != sel_iy)
			
			#Increment Counter
			B    += 1
		
		# Compute average
		train_err  = float(train_err)  / len(train_ix)
		train_loss = float(train_loss) / len(train_ix)
		
		# Print
		print("                                 **** TRAIN ERR: {:12.6f}    LOSS: {:12.6f} ****".format(train_err*100.0, train_loss));
		
		# Timing end
		te = time.time()
		print("                                 **** EPOCH {:3d} TIME: {:12.6f} seconds ****".format(e, te-ts));
		
		dumpSVs(DEFAULT_FILEPATH, SV)
	
	print("Done!")
	pdb.set_trace()

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
