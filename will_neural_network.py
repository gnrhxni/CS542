#!/usr/bin/python

import os
import re
import sys
import numpy
import pybrain
import math
from nettalk_data import *
from constants import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer 
from pybrain.tools.shortcuts import buildNetwork

print("Setting up network")
ITERATIONS = 10
NUMINPUTS = 29*7
neural_network = buildNetwork (
    NUMINPUTS, 
    80, 
    NUMOUTPUTS, 
    bias=True, 
    outputbias=True, 
    hiddenclass=SigmoidLayer, 
    outclass=SigmoidLayer)
newWeights = np.random.uniform(-0.3, 0.3, len(neural_network.params))
print(newWeights)
neural_network._setParameters(newWeights)
print(neural_network.params)
print("Setting up trainer")
trainer = BackpropTrainer(
    neural_network, 
	None, 
    learningrate=1.0, 
    verbose=True, 
    batchlearning=True, 
    weightdecay=0.0)
	  
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
    print("Phoneme accuracy = ", 1-np.mean(phoneme_error), " Stress accuracy = ", 1-np.mean(stress_error))
	
def main():
    for i in range(ITERATIONS):
        trainNetwork()
		
main()
