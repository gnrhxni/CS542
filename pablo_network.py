#!/usr/bin/python

from nettalk_data import *
import re
import os
import sys
import numpy
import pickle
import pprint
#import pdb
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LinearLayer, SigmoidLayer, BiasUnit 
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from sigmoidsparselayer import SigmoidSparseLayer

lrate = float(sys.argv[1]);
wdecay = float(sys.argv[2]);
print (lrate, wdecay);


def buildNet():

    inLayer = LinearLayer(NUMINPUTS, name='in')
    hiddenLayer = SigmoidSparseLayer(dim=80, beta=0.001, sparsity=0.1, name='hidden')
    #hiddenLayer = SigmoidLayer(dim=80, name='hidden')
    outLayer = SigmoidLayer(NUMOUTPUTS, name='out')
    biasUnit = BiasUnit(name='bias')

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    bias_to_hidden = FullConnection(biasUnit, hiddenLayer)
    bias_to_out = FullConnection(biasUnit, outLayer)

    tosave = [ inLayer, hiddenLayer, outLayer, biasUnit, in_to_hidden, hidden_to_out, bias_to_hidden, bias_to_out ];

    return tosave


if (len(sys.argv) <= 3):
    saved = buildNet()
else:
    saved = pickle.load(open(sys.argv[3], "rb"));

pickle.dump( saved, open( "pablosemptynet.p", "wb" ) )


net = FeedForwardNetwork(name='mynet');

net.addInputModule(saved[0])
net.addModule(saved[1])
net.addOutputModule(saved[2])
net.addModule(saved[3])
net.addConnection(saved[4])
net.addConnection(saved[5])
net.addConnection(saved[6])
net.addConnection(saved[7])

net.sortModules()

trainer = BackpropTrainer(net, None, learningrate=lrate, verbose=False, batchlearning=True, weightdecay=wdecay)                                        
stressErrors=list();
phonemeErrors=list();


for cycle in range(100):
	datafile = 'top1000.data';
	for entry in dictionary(datafile):
		#print("working on", entry);
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
                #pdb.set_trace();
		err = trainer.train();
		#print(err, " ", entry);
	print("accuracy: phonemes %.3f stresses %.3f" % (1 - np.mean(phonemeErrors), 1 - np.mean(stressErrors)) );

    
#accuracy is a vector with one element in {0,1} for each letter i
#that we have #trained so far. 
#make that two vectors, one for phoneme and one for stress.

