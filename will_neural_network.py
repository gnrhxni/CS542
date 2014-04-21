#!/usr/bin/python

import os
import re
import sys
import numpy
import pybrain
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

print("Setting up network")
ITERATIONS = 10
modules = buildModules(NUMINPUTS, 80, NUMOUTPUTS)
neural_network = buildnet(modules)
#IMPORTANT: IF YOU WANT TO SET YOUR WEIGHTS TO -0.3 to 0.3, please use the following 4 lines
newWeights = np.random.uniform(-0.3, 0.3, len(neural_network.params))
#print(newWeights)
neural_network._setParameters(newWeights)
#print(neural_network.params)
print("Setting up trainer")
trainer = BackpropTrainer(
    neural_network, 
	None, 
    learningrate=1.0, 
    verbose=False, 
    batchlearning=True, 
    weightdecay=0.0)


def testOneWord(word, output=None):
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

def testWords(inputfile):
    phoneme_error = list()
    stress_error = list()
    #loop through each word in our data, treating each one as a seperate dataset
    for word in dictionary(inputfile):
        (pherrors, serrors) = testOneWord(word);
        phoneme_error.extend(pherrors);
        stress_error.extend(serrors);
    print("Generalization: phoneme %.3f stress %.3f" % ( 1-np.mean(phoneme_error), 1-np.mean(stress_error)) )
    
	  
def trainNetwork():
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
        (pherrors, serrors) = testOneWord(word, output);
        phoneme_error.extend(pherrors);
        stress_error.extend(serrors);
    print("Accuracy: phoneme %.3f stress %.3f" % ( 1-np.mean(phoneme_error), 1-np.mean(stress_error)) )

	
def main():
    for i in range(ITERATIONS):
        start=time.time()
        trainNetwork()
        trained = time.time()
        testWords('nettalk.data')
        tested = time.time()
        print("time to train ", trained - start, " to test ", tested - trained)
        

main()
