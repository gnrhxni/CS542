#!/usr/bin/python

import pickle
import numpy as np
import glob

stems = 'weights.forgiving_1_1 weights.forgiving_0_0'.split();
for s in stems:
    files = glob.glob('%s*' % s);
    a = np.zeros(0);
    for f in files:
        w = pickle.load(open(f));
        a = np.append(a,w);
    h = np.histogram(a,range(-20,20,2));
    print("histogram for %s" % s);
    print(h);
        
