__author__ = 'Pablo Alvarez, palvarez@palvarez.net'

from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.tools.functions import sigmoid
import numpy


class SigmoidSparseLayer(SigmoidLayer):
    """Layer implementing the sigmoid squashing function
    and a sparsity constraint"""

    def __init__(self, dim, sparsity=0.1, beta=1, resetAfterTraining=True, name=None):
        SigmoidLayer.__init__(self, dim, name);
        self.r = sparsity;
        self.beta = beta;
        self.resetAfterTraining= resetAfterTraining;
        self.resetAverage();

    def resetAverage(self):
        self.saved = numpy.zeros(self.dim, numpy.float64 );
        self.numsaved = 0;

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = sigmoid(inbuf)
        self.saved += outbuf;
        self.numsaved += 1;
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        numpy.set_printoptions(precision=3,suppress=True,linewidth=1000)
        #print("outerr", outerr);
        if (self.beta > 0 and self.numsaved > 0):
            avg = self.saved / self.numsaved;
            #print(avg);
            r = self.r;
            for i in range(len(avg)):
                if (avg[i] > 0.01 and avg[i] < 0.99):
                    outerr[i] += self.beta * (-r/avg[i] + (1-r)/(1-avg[i]));
        #print("outerr changed", outerr);
        inerr[:] = outbuf * (1 - outbuf) * outerr;
        if (self.resetAfterTraining): self.resetAverage();

    def sparsityCost(self):
        avg = saved / numsaved; 
        r = self.r;
        kl = r * log(r/avg) + (1-r) * log( (1-r) / (1-avg) );
        costSparse = sum(beta * kl);
        return costSparse
    
        
