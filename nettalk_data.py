#/usr/bin/python

import re
import sys
from collections import namedtuple
import numpy as np
import csv

from constants import (
    topK,
    WEIRD,
    FOREIGN,
    MINSTRESS,
    NUMOUTPUTS,
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
        phoneme = entry.phonemes[i];
        features = articFeatures[phoneme];
        for f in features: 
            ret[i][f] = 1;
        features = stressFeatures[entry.stress[i]];
        for f in features: 
            ret[i][MINSTRESS + f] = 1;
        if (1 == entry.flag): 
            ret[i][WEIRD] = 1;
        elif (2 == entry.flag): 
            ret[i][FOREIGN] = 1;
        print(phoneme, " ", ret[i]);
    return ret;


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

    
