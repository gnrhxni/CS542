#!/usr/bin/python

import os
import re
import sys
import numpy
import pybrain
import pickle
import math
import time
import random
from nettalk_data import *
from constants import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LinearLayer, SigmoidLayer, BiasUnit 
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from sigmoidsparselayer import SigmoidSparseLayer
from pybrain.tools.shortcuts import buildNetwork
from prevent_overtraining import PreventOverTrainer

from nettalk_modules import *


WORDSTRAINED=100000;

def setup(hidden=80, hidden2=0, forgivingHidden=False, forgivingOut=False):
    print("Setting up network")
    modules = buildModules(NUMINPUTS, hidden, NUMOUTPUTS, hidden2=hidden2, forgivingHidden=forgivingHidden, forgivingOut=forgivingOut)
    neural_network = buildnet(modules)
#IMPORTANT: IF YOU WANT TO SET YOUR WEIGHTS TO -0.3 to 0.3, please use the following 4 lines
    newWeights = np.random.uniform(-0.3, 0.3, len(neural_network.params))
    #pickle.dump(newWeights, open("weights.p","wb"))
    #newWeights = pickle.load(open("weights.p","rb"))
#print(newWeights)
    neural_network._setParameters(newWeights)
#print(neural_network.params)
    print("Setting up trainer")
    return (neural_network, modules)

def testOneWord(neural_network, word, output=None):
   """ Return (phoneme_error, stress_error) lists

   The lists are parallel and indexed by letter position
   in the input word, which must be a DictionaryEntry.
   if output is missing, it is calculated, but it can also
   be passed in to save time if we already have it"""
   phoneme_error = list()
   stress_error = list()
   if (None == output): output = outputUnits(word)
   char_pos = 0
   for letter in wordstream(input_entries = (word,)):
            #now convert these 7-character sequences into binary
            for binary_input in convertToBinary(letter):
                #determine the corresponding correct output and add the sample to the dataset
                binary_output = output[char_pos]
                network_output = neural_network.activate(binary_input)
                phoneme = word.phonemes[char_pos]
                stress = word.stress[char_pos]
                calculated_phoneme = closestByDotProduct(network_output[:MINSTRESS], articFeatures)
                calculated_stress = closestByDotProduct(network_output[MINSTRESS:], stressFeatures)
                phoneme_error.append(bool(phoneme != calculated_phoneme))
                stress_error.append(bool(stress != calculated_stress))
                char_pos = char_pos + 1
   return (phoneme_error, stress_error)

def testWords(neural_network, inputfile):
    phoneme_error = list()
    stress_error = list()
    #loop through each word in our data, treating each one as a seperate dataset
    for word in dictionary(inputfile):
        (pherrors, serrors) = testOneWord(neural_network, word);
        phoneme_error.extend(pherrors);
        stress_error.extend(serrors);
    print("Generalization: phoneme %.3f stress %.3f" % ( 1-np.mean(phoneme_error), 1-np.mean(stress_error)) )
    return ( 1-np.mean(phoneme_error), 1-np.mean(stress_error))
    
	  
def trainNetwork(neural_network, trainer, trainfile, testfile, outfile, testSkip=1000):
    ret = ([], [], [])
    #loop through each word in our data, treating each one as a seperate dataset
    for word in dictionary(trainfile):
        output = outputUnits(word)
        ds = SupervisedDataSet(NUMINPUTS, NUMOUTPUTS)
        char_pos = 0
        #loop through each letter in the word, and center it in a 7-character sequence
        for letter in wordstream(input_entries = (word,)):
            #now convert these 7-character sequences into binary
            for binary_input in convertToBinary(letter):
                #determine the corresponding correct output and add the sample to the dataset
                binary_output = output[char_pos]
                ds.addSample(binary_input, binary_output)
                char_pos+= 1
        trainer.setData(ds)
        err = trainer.train()
        trainNetwork.counter += 1
        if (0 == trainNetwork.counter % testSkip):
            testerror = testWords(neural_network, testfile);
            ret[0].append(trainNetwork.counter);
            ret[1].append(testerror[0]);
            ret[2].append(testerror[1]);
            outfile.write("%d %.3f %.3f\n" % (trainNetwork.counter, testerror[0], testerror[1]))
            outfile.flush();
    return ret;
trainNetwork.counter=0

def main():
  experiment = [];
  hidden=80
  hidden2=0
  lrate=0.4
  beta=0
  r=0.5
  testSkip=1000
  train="firsthalf.disordered.data";
  test="secondhalf.disordered.data";
  for (iteration) in (1,2):
    for lrate in (0.8,0.4,1.6,0.2,0.1,3.2):
      (net, modules) = setup(hidden)
      trainer = BackpropTrainer( net, None, learningrate=lrate, verbose=False, batchlearning=True, weightdecay=0.0)
      fname = 'phonemes_lrate_%.2f.%d' % (lrate, int(time.time()))
      outfile = open(fname,'w')
      trainNetwork.counter=0
      while trainNetwork.counter < WORDSTRAINED:
        trainerror = trainNetwork(net, trainer, train, test, outfile, testSkip=testSkip)
        experiment.append(trainerror) 
      weights = net.params
      pickle.dump(weights, open("weights.%s" % fname,"wb"))
  for i in experiment:
     print i;

if __name__ == '__main__':
   main()
