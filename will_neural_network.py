#!/usr/bin/python

import os
import re
import sys
import numpy
import pybrain
from nettalk_data import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer 
from pybrain.tools.shortcuts import buildNetwork

print("Setting up network")
ITERATIONS = 10;

neural_network = buildNetwork (
    NUMINPUTS, 
    80, 
    NUMOUTPUTS, 
    bias=True, 
    outputbias=True, 
    hiddenclass=SigmoidLayer, 
    outclass=SigmoidLayer)

print("Setting up trainer")
trainer = BackpropTrainer(
    neural_network, 
	None, 
    learningrate=1.0, 
    verbose=True, 
    batchlearning=True, 
    weightdecay=0.0)
	
#pablos dot product function
def closestByDotProduct(features, compareDict):
	maxcos = 0;
	best = '';
	for k in compareDict.keys():
		f = np.zeros(len(features));
		for on in compareDict[k]: f[on] = 1;
        #Actually, we should also divide below by sqrt(norm(features)), but we don't care because we are just trying to compare across possible fs, and that's a constant. We should save the f vectors and the norms for all the phonemes in some dict so we don't have to recalculate them.
		cos = np.dot(f,features) / sqrt(np.dot(f,f));
		#print("%s\n%s %.2f %s" % ((features*10).astype(int), (f*10).astype(int), cos, k));
		if (cos > maxcos):
			maxcos = cos;
			best = k;
	return best;
        
def trainNetwork():
    phoneme_error = list()
    stress_error = list()
    #loop through each word in our data, treating each one as a seperate dataset
    for word in dictionary(defaultdatafile)
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
                network_output = neural_network.activate(binary_input)
                phoneme = word.phonemes[char_pos]
                stress = word.stress[char_pos]
                calculated_phoneme = closestByCrossProduct(network_output[:MINSTRESS], articFeatures])
                calculated_stress = closestByCrossProduct(network_output[MINSTRESS:], stressFeatures])
                if phoneme != calculated_phoneme:
                    phoneme_error.append(0)
                else:
                    phoneme_error.append(1)
				
                if stress != calculated_stress:
                    stress_error.append(0)
                else:
                    stress_error.append(1)
                char_pos = char_pos + 1
        trainer.setData(ds)
        err = trainer.train()
    print("Phoneme accuracy = ", np.mean(phoneme_error), " Stress accuracy = ", np.mean(stress_error))
	
def main():
    for i in range(ITERATIONS):
        trainNetwork()
		
main()