#/usr/bin/python

import re
import sys
from collections import namedtuple
import numpy as np
import csv
from math import sqrt

from constants import (
    topK,
    MINSTRESS,
    NUMOUTPUTS,
    NUMINPUTS,
    letterToPos,
    topKDatafile,
    articFeatures,
    stressFeatures,
    defaultdatafile,
    articFeatureNames,
    stressFeatureNames
)

Dictionary_Element = namedtuple("Dictionary_Element", 
                                "word phonemes stress flag")


def testit():
    for entry in dictionary():
        print(entry.word); 
        outarray = outputUnits(entry);
        print(outarray);
        sys.exit(1);


def dictionary(datafile=defaultdatafile):
    with open(datafile) as data_fp:
        for line in data_fp:
            match = re.match(r'([a-z]+)\t(\S+)\t([\d<>]+)\t(\d+)', line)
            if match:
                yield Dictionary_Element._make(match.groups())


def wordstream(windowsize=7, input_entries=None, padchar=' '):
    """Note: middle of each window is (windowsize/2)+1, since python
    automatically floors uneven integer division.
    """
    even = bool(windowsize % 2 == 0)
    if even:
        lmargin, rmargin = windowsize/2, windowsize/2
    if not even:
        lmargin, rmargin = windowsize/2, (windowsize/2)+1

    if not input_entries:
        input_entries = dictionary()

    for entry in input_entries:
        word = entry.word
        ret = list()
        for i in range(len(word)):
            chunk = word[max(i-lmargin,0):i+rmargin]
            lpad = -min(i-lmargin, 0)
            rpad = windowsize - len(chunk) - lpad
            ret.append( 
                padchar*(lpad) + chunk + padchar*(rpad)
            )
                
        yield ret

def binarystream(**kwds):
    return iter( 
        list( convertToBinary(wordsalad) )
        for wordsalad in wordstream(**kwds) 
    )

def createFeatureTable(filename):
    features = dict();
    featureToUnit = dict();
    csvfile =  open(filename, 'r');
    reader = csv.reader(csvfile, delimiter=',', quotechar='"');
    for row in reader: 
        if (1 != len(row[0])): continue;
        for i in range(3,len(row)):
            if (row[i] not in featureToUnit and len(row[i])):
                featureToUnit[row[i]] = len(featureToUnit);
    print(featureToUnit);
    csvfile.seek(0);
    reader = csv.reader(csvfile, delimiter=',', quotechar='"');
    for row in reader: 
        grapheme = row[0];
        if (1 != len(grapheme)): continue;
        if (grapheme not in features):
          features[grapheme] = [];
        for i in range(3,len(row)):
            if (len(row[i])):
                intfeature = featureToUnit[row[i]];
                features[grapheme].append(intfeature);
    return (features, len(featureToUnit));   


def outputUnits(entry):
    ret = np.zeros((len(entry.word), NUMOUTPUTS), np.int8)
    for i in range(len(entry.word)):
        parray = (articFeatures[entry.phonemes[i]])[0]
        sarray = stressFeatures[entry.stress[i]][0]
        ret[i] = np.hstack((parray,sarray))
    return ret;



def closestByDotProductSlow(features, positionalFeatureDict):
	maxcos = 0;
	best = '';
	for k in positionalFeatureDict.keys():
		f = np.zeros(len(features));
		for on in positionalFeatureDict[k]: f[on] = 1;
        #Actually, we should also divide below by sqrt(norm(features)), but we don't care because we are just trying to compare across possible fs, and that's a constant. We should save the f vectors and the norms for all the phonemes in some dict so we don't have to recalculate them.
		cos = np.dot(f,features) / sqrt(np.dot(f,f));
		#print("%s\n%s %.2f %s" % ((features*10).astype(int), (f*10).astype(int), cos, k));
		if (cos > maxcos):
			maxcos = cos;
			best = k;
	return best;
        
def closestByDotProduct(features, vectorDict):
	maxcos = 0;
	best = '';
	for k,v in vectorDict.iteritems():
        #Actually, we should also divide below by sqrt(norm(features)), but we don't care because we are just trying to compare across possible fs, and that's a constant. We should save the f vectors and the norms for all the phonemes in some dict so we don't have to recalculate them.
		cos = np.dot(v[0],features) / v[1];
		#print("%s\n%s %.2f %s" % ((features*10).astype(int), (f*10).astype(int), cos, k));
		if (cos > maxcos):
			maxcos = cos;
			best = k;
	return best;
        
def createVectorDict(featureDict, length):
    v = dict();
    for k in featureDict.keys():
        f = np.zeros(length);
        for on in featureDict[k]: f[on] = 1;
        norm = sqrt(np.dot(f,f));
        v[k] = (f, norm);
    return v;     	

def convertToBinary(words=None):
    for word in words:
        representation = list()
        word = word.lower()
        for letter in word:
            for i in range(29):
                if i == letterToPos[letter]:
                    representation.append(1)
                else:
                    representation.append(0)
        yield representation

    
