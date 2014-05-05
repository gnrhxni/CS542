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

WORDSTRAINED=25000
children = []
master = None

def setup(numNetworks = 1, hidden=80, hidden2=0, forgiving=False):
    child_networks = []
    for i in range(numNetworks):
        neural_network = buildnet(buildModules(NUMINPUTS, hidden, NUMOUTPUTS, hidden2=hidden2, forgiving=forgiving))
        newWeights = np.random.uniform(-0.3, 0.3, len(neural_network.params))
        neural_network._setParameters(newWeights)
        child_networks.append(neural_network)
    master_network = buildnet(buildModules(NUMOUTPUTS*numNetworks, hidden, NUMOUTPUTS, hidden2=hidden2, forgiving=forgiving))
    newWeights = np.random.uniform(-0.3, 0.3, len(neural_network.params))
    master_network._setParameters(newWeights)
    return (child_networks, master_network)

def createDatasetFromWord(word):
    output = outputUnits(word)
    ds = SupervisedDataSet(NUMINPUTS, NUMOUTPUTS)
    char_pos = 0
    #loop through each letter in the word, and center it in a 7-character sequence
    for letter in wordstream(input_entries = (word,)):
        for binary_input in convertToBinary(letter):
            binary_output = output[char_pos]
            ds.addSample(binary_input, binary_output)
            char_pos+= 1
    return ds
def createMasterDataset(words, networks):
    ds = SupervisedDataSet(NUMOUTPUTS*len(networks), NUMOUTPUTS)
    for word in words:
        output = outputUnits(word)
        char_pos = 0
        #loop through each letter in the word, and center it in a 7-character sequence
        for letter in wordstream(input_entries = (word,)):
            for binary_input in convertToBinary(letter):
                final_input = []
                binary_output = output[char_pos]
                for (network, trainer) in networks:
                    final_input = final_input + network.activate(binary_input).tolist()
                ds.addSample(final_input, binary_output)
                
                char_pos+= 1
    return ds
def testOneWord(children, master_network, word, output=None):
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
                final_input = []
                #determine the corresponding correct output and add the sample to the dataset
                binary_output = output[char_pos]
                for (network, trainer) in children:
                    final_input = final_input + network.activate(binary_input).tolist()
                network_output = master_network.activate(final_input)
                phoneme = word.phonemes[char_pos]
                stress = word.stress[char_pos]
                calculated_phoneme = closestByDotProduct(network_output[:MINSTRESS], articFeatures)
                calculated_stress = closestByDotProduct(network_output[MINSTRESS:], stressFeatures)
                phoneme_error.append(bool(phoneme != calculated_phoneme))
                stress_error.append(bool(stress != calculated_stress))
                char_pos = char_pos + 1
   return (phoneme_error, stress_error)

def testWords(children, master_network, inputfile):
    phoneme_error = list()
    stress_error = list()
    #loop through each word in our data, treating each one as a seperate dataset
    for word in dictionary(inputfile):
        (pherrors, serrors) = testOneWord(children, master_network, word);
        phoneme_error.extend(pherrors);
        stress_error.extend(serrors);
    print("Generalization: phoneme %.3f stress %.3f" % ( 1-np.mean(phoneme_error), 1-np.mean(stress_error)) )
    return ( 1-np.mean(phoneme_error), 1-np.mean(stress_error))
    
def trainNetwork(children, master, trainfile, testfile, outfile, testSkip=1000):
    ret = ([], [], [])
    numChildren = len(children)
    cycle = 0
    #loop through each word in our data, treating each one as a seperate dataset
    curWords = []
    for word in dictionary(trainfile):
        (neural_network, trainer) = children[cycle]
        trainer.setData(createDatasetFromWord(word))
        err = trainer.train()
        cycle += 1
        curWords.append(word)
        if cycle == numChildren:
            cycle = 0
            (master_network, master_trainer) = master
            master_dataset = createMasterDataset(curWords, children)
            master_trainer.setData(master_dataset)
            err = master_trainer.train()
            trainNetwork.counter += numChildren
            curWords = []
        if (0 > trainNetwork.counter % testSkip):
            (master_network, master_trainer) = master
            testerror = testWords(children, master_network, testfile);
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
  numNetworks = 4
  testSkip=1000
  f = open('nettalk.data','r');
  f1 = open('temptrain','w'); 
  f2 = open('temptest','w'); 
  l1 = set()
  l2 = set()
# Put all the lines in f in random order into f1 and f2
# in the proportion described by "proportion"
  for l in f.readlines():
    if (random.random() < 0.5): l1.add(l)
    else: l2.add(l);
  for l in l1: f1.write(l);
  for l in l2: f2.write(l);
  f.close(); f1.close(); f2.close();
  for (train, test) in (('temptrain','temptest'),):
   (child_networks, master_network) = setup(numNetworks, hidden, hidden2)
   for child_network in child_networks:
       child_trainer = BackpropTrainer(child_network, None, learningrate=lrate, verbose=True, batchlearning=True, weightdecay=0.0)
       children.append((child_network, child_trainer))
   master_trainer = BackpropTrainer(master_network, None, learningrate=lrate, verbose=True, batchlearning=True, weightdecay=0.0)
   master = (master_network, master_trainer) 
   fname = 'numChildren_%.1f.%d' % (numNetworks, int(time.time()))
   outfile = open(fname,'w')
   trainNetwork.counter=0
   while trainNetwork.counter < WORDSTRAINED:
       trainerror = trainNetwork(children, master, train, test, outfile, testSkip=testSkip)
       experiment.append(trainerror) 


if __name__ == '__main__':
   main()
