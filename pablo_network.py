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

stressErrors=list();
phonemeErrors=list();

for cycle in range(1000):
	datafile = 'top1000.data';
	for entry in dictionary(datafile):
		print("working on", entry);
		outmatrix = outputUnits(entry);
		lpos = 0;
		ds = SupervisedDataSet(NUMINPUTS, NUMOUTPUTS);
		for letterContexts in wordstream(input_entries = (entry,)):
			#print("letterContexts", letterContexts);
			for inarray in convertToBinary(letterContexts):
				outarray = outmatrix[lpos];
		#print("inarray",inarray);
		#print("outarray",outarray); 
	#print("inlen %d outlen %d" % (len(inarray), len(outarray)));
				ds.addSample(inarray, outarray);
				observed = net.activate(inarray);
				phoneme = entry.phonemes[lpos];
				observedPhoneme = closestByDotProduct(observed[:MINSTRESS], articFeatures);
				phonemeErrors.append(bool(phoneme != observedPhoneme));
				stress = entry.stress[lpos];
				observedStress = closestByDotProduct(observed[MINSTRESS:], stressFeatures);
				stressErrors.append(bool(stress != observedStress));
				lpos += 1
		trainer.setData(ds);
		err = trainer.train();
		print(err, " ", entry);
	print("accuracy: phonemes %.3f stresses %.3f" % (1 - np.mean(phonemeErrors), 1 - np.mean(stressErrors)) );

    
#accuracy is a vector with one element in {0,1} for each letter i
#that we have #trained so far. 
#make that two vectors, one for phoneme and one for stress.

