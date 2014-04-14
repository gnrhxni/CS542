#!/usr/bin/python

from nettalk_data import *
import re
import os
import sys
import numpy
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer 
from pybrain.datasets import SupervisedDataSet

net = buildNetwork (NUMINPUTS, 80, NUMOUTPUTS, bias=True, outputbias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)
trainer = BackpropTrainer(net, None, learningrate=1.0, verbose=True, batchlearning=True, weightdecay=0.0)                                        
for cycle in range(5):
  datafile = 'top1000.data';
  for entry in dictionary(datafile):
    #print("working on", entry);
    outmatrix = outputUnits(entry);
    whichletter = 0;
    ds = SupervisedDataSet(NUMINPUTS, NUMOUTPUTS);
    for letterContexts in wordstream(input_entries = (entry,)):
      #print("letterContexts", letterContexts);
      for inarray in convertToBinary(letterContexts):
      	outarray = outmatrix[whichletter];
	whichletter += 1
      	#print("inarray",inarray);
      	#print("outarray",outarray); 
	#print("inlen %d outlen %d" % (len(inarray), len(outarray)));
	ds.addSample(inarray, outarray);
    trainer.setData(ds);
    err = trainer.train();
    print(err, " ", entry);
    
  #create input vectors
  #create output vectors

  #trainer.setData(ds)
  #trainer.train()
  #for each letter:
  #  out = net.activate(input);
  #  for each phoneme and each stress separately:
  #        angle = dotprod(out, output)/(norm(out)*norm(output))
  #    compare angles, pick the smallest.
  #    add to accuracy measure.
#accuracy is a vector with one element in [0,1] for each word that we have
#trained so far that is the average of the accuracy for each letter. Actually 
#make that two vectors, one for phoneme and one for stress.
#endeachword
#endeachtrainingcycle

