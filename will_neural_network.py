#!/usr/bin/python

import os
import re
import sys
import numpy
import pybrain
import pickle
import math
import time
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

from nettalk_modules import *


ITERATIONS = 20

def setup():
    print("Setting up network")
    modules = buildModules(NUMINPUTS, 80, NUMOUTPUTS)
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
    
	  
def trainNetwork(neural_network, trainer):
    phoneme_error = list()
    stress_error = list()
    #loop through each word in our data, treating each one as a seperate dataset
    for word in dictionary('top1000.data'):
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
        (pherrors, serrors) = testOneWord(neural_network, word, output);
        phoneme_error.extend(pherrors);
        stress_error.extend(serrors);
    print("Accuracy: phoneme %.3f stress %.3f" % ( 1-np.mean(phoneme_error), 1-np.mean(stress_error)) )
    return ( 1-np.mean(phoneme_error), 1-np.mean(stress_error))

	
def main():
  experiment = []
  for beta in (0,0,0.0003,0.0003):
   for r in (0.4, 0.6, 0.8):
    (net, modules) = setup()
    trainer = BackpropTrainer( net, None, learningrate=1.0, verbose=False, batchlearning=True, weightdecay=0.0)
    modules['hidden'].beta = beta
    modules['hidden'].r = r
    print ("beta ", beta, " r ", r)
    for i in range(ITERATIONS):
        data = [ beta, r, i ];
        #start=time.time()
        trainerror = trainNetwork(net, trainer)
        data.extend(trainerror)
        #trained = time.time()
        if (0==(i+1)%5): 
            testerror = testWords(net, 'nettalk.data')
            data.extend(testerror)
        #tested = time.time()
        #print("time to train ", trained - start, " to test ", tested - trained)
        experiment.append(data)
  for i in experiment:
     print i;

if __name__ == '__main__':
   main()
