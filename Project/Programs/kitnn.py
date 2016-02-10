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






###############################################################################
# Implementations of the script's "verbs".
#

#
# Annotate dataset with bounding boxes.
#

def verb_annotatebbds(argv):
	print argv

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

def verb_train(argv):
	print argv

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
