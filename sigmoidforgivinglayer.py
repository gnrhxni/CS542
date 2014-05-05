__author__ = 'Pablo Alvarez, palvarez@palvarez.net'

from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.tools.functions import sigmoid
import numpy
from pprint import pprint


class SigmoidForgivingLayer(SigmoidLayer):
    """Layer implementing the sigmoid squashing function
    and not propagating any errors <= 0.1"""

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = sigmoid(inbuf)
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        useit = abs(outerr[:]) > 0.1;
        inerr[:] = outbuf * (1 - outbuf) * outerr;
        inerr = useit * inerr;

        
