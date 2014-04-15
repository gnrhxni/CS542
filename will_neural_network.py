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

def trainNetwork():
    phoneme_error = list()
	stress_error = list()
    #loop through each word in our data, treating each one as a seperate dataset
	for word in dictionary('top1000.data')
		output = outputUnits(word)
		ds = SupervisedDataSet(NUMINPUTS, NUMOUTPUTS)
		char_pos = 0
		#loop through each letter in the word, and center it in a 7-character sequence
		for letter in wordstream(input_entries = (word,)):
		    #now convert these 7-character sequences into binary
			for binary_input in convertToBinary(letter):
				#determine the corresponding correct output and add the sample to the dataset
				binary_output = output[char_pos];
				ds.addSample(binary_input, binary_output)
				network_output = neural_network.activate(binary_input)
				phoneme = word.phonemes[char_pos]
				stress = word.stress[char_pos]
				#calculate phoneme based on network outpu using dot product (still need to do)
				#calculate stree based on network outpit using dot product (still need to do)
				if phoneme != calculated_phoneme:
					phoneme_error.append(1)
				else:
					phoneme_error.append(0)
				
				if stress != calculated_stress:
					stress_error.append(1)
				else:
					stress_error.append(0)
				char_pos = char_pos + 1
		trainer.setData(ds);
		err = trainer.train();
	print("Phoneme accuracy = ", (1 - np.mean(phoneme_error)), " Stress accuracy = ", (1 - np.mean(stress_error)));
	
def main():
    for i in range(ITERATIONS):
	    trainNetwork()
		
main()